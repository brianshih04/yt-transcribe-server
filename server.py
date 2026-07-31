"""
YouTube Audio Transcription Service (async job mode)
接收 YouTube URL → yt-dlp 抓音訊 → MOSS-Transcribe (:8006) → 回傳逐字稿

POST /transcribe      → 立刻回 {job_id}（背景處理）
GET  /job/{job_id}    → 查進度 / 取結果
GET  /health
"""

import os
import re
import json
import uuid
import shutil
import tempfile
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── 配置 ──────────────────────────────────────────────

MOSS_URL = os.environ.get("MOSS_URL", "http://localhost:8006/v1/audio/transcriptions")
PORT = int(os.environ.get("PORT", "8503"))
MOSS_CHUNK_S = int(os.environ.get("MOSS_CHUNK_S", "300"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yt-transcribe")

# ── Job Store ────────────────────────────────────────

_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()

# ── FastAPI ──────────────────────────────────────────

app = FastAPI(title="YouTube Transcription Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    url: str
    language: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/captions/{video_id}")
async def get_captions(video_id: str, lang: Optional[str] = None):
    """透過 Innertube ANDROID API 取得 YouTube 字幕。"""
    logger.info("Captions request: %s", video_id)

    # Step 1: Innertube player API (ANDROID client)
    try:
        resp = requests.post(
            "https://www.youtube.com/youtubei/v1/player?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
            json={
                "context": {"client": {
                    "clientName": "ANDROID",
                    "clientVersion": "20.10.38",
                    "androidSdkVersion": 30,
                }},
                "videoId": video_id,
            },
            headers={
                "Content-Type": "application/json",
                "User-Agent": "com.google.android.youtube/20.10.38",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Innertube HTTP {resp.status_code}")
        data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Innertube error: {e}")

    tracks = (data.get("captions") or {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
    if not tracks:
        raise HTTPException(404, "No caption tracks")

    # 選字幕軌
    target = tracks[0]
    if lang:
        for t in tracks:
            if t.get("languageCode", "").startswith(lang):
                target = t
                break

    base_url = target["baseUrl"].replace("\\u0026", "&")
    logger.info("Track: %s, URL: %s...", target.get("languageCode"), base_url[:80])

    # Step 2: fetch timedtext
    for fmt in ["srv3", "json3", ""]:
        url = f"{base_url}&fmt={fmt}" if fmt else base_url
        try:
            r = requests.get(url, headers={
                "User-Agent": "com.google.android.youtube/20.10.38",
            }, timeout=15)
            logger.info("fmt=%s: HTTP %d, len=%d", fmt or "none", r.status_code, len(r.text))
            if r.text.strip():
                segments = _parse_timedtext(r.text, fmt)
                if segments:
                    logger.info("Captions OK: %d segments", len(segments))
                    return {"segments": segments, "language": target.get("languageCode", "unknown")}
        except Exception as e:
            logger.warning("fmt=%s error: %s", fmt or "none", e)
            continue

    raise HTTPException(502, "All timedtext formats returned empty")


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    url = req.url.strip()
    if not url or "youtube.com" not in url:
        raise HTTPException(400, "Invalid URL")

    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(400, "Cannot extract video ID")

    job_id = uuid.uuid4().hex[:12]

    with _job_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "排隊中...",
            "result": None,
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, url, video_id, req.language),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "video_id": video_id}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


# ── Background Job ───────────────────────────────────

def _update_job(job_id: str, **kwargs):
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_job(job_id: str, url: str, video_id: str, language: Optional[str]):
    work_dir = tempfile.mkdtemp(prefix="yt_trans_")
    try:
        _update_job(job_id, status="processing", progress=5, message="下載音訊中...")

        # Step 1: 下載
        raw_audio = os.path.join(work_dir, "raw.webm")
        audio_path = os.path.join(work_dir, "audio.wav")

        yt_dlp = os.path.join(os.path.dirname(__file__), "venv", "bin", "yt-dlp")
        result = subprocess.run(
            [yt_dlp, "-f", "bestaudio/best", "--no-playlist", "-o", raw_audio, url],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise Exception(f"Download failed: {result.stderr[-200:]}")

        raw_files = [f for f in Path(work_dir).glob("raw.*") if f.stat().st_size > 0]
        if not raw_files:
            raise Exception("Audio file not found")
        raw_file = str(raw_files[0])

        # Step 2: ffmpeg 16kHz mono
        _update_job(job_id, progress=15, message="轉換音訊格式...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_file,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path],
            capture_output=True, text=True, timeout=120,
        )

        # Step 3: 分段送 MOSS
        dur_out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True,
        )
        total_duration = float(dur_out.stdout.strip() or "0")
        logger.info("[%s] Duration: %.0fs", job_id, total_duration)

        all_segments = []
        detected_lang = "unknown"

        if total_duration <= MOSS_CHUNK_S:
            _update_job(job_id, progress=30, message="語音辨識中...")
            all_segments, detected_lang = _moss_transcribe(audio_path, language)
        else:
            num_chunks = int(total_duration // MOSS_CHUNK_S) + 1
            for ci in range(num_chunks):
                chunk_start = ci * MOSS_CHUNK_S
                chunk_end = min((ci + 1) * MOSS_CHUNK_S, total_duration)
                if chunk_start >= total_duration:
                    break

                progress = 20 + int((ci / num_chunks) * 70)
                _update_job(job_id, progress=progress,
                            message=f"語音辨識中... 第 {ci+1}/{num_chunks} 段")

                chunk_path = os.path.join(work_dir, f"chunk_{ci:03d}.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path,
                     "-ss", str(chunk_start), "-to", str(chunk_end),
                     "-c", "copy", chunk_path],
                    capture_output=True, text=True, timeout=60,
                )
                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                    continue

                segments, detected_lang = _moss_transcribe(chunk_path, language)
                for seg in segments:
                    seg["start"] = round(seg["start"] + chunk_start, 2)
                    all_segments.append(seg)
                logger.info("[%s] Chunk %d/%d: %d segs", job_id, ci+1, num_chunks, len(segments))

        logger.info("[%s] Done: %d segments", job_id, len(all_segments))
        _update_job(
            job_id,
            status="done",
            progress=100,
            message="完成",
            result={
                "success": True,
                "videoId": video_id,
                "segments": all_segments,
                "language": detected_lang,
                "duration": total_duration,
            },
        )

    except Exception as e:
        logger.error("[%s] Failed: %s", job_id, e)
        _update_job(job_id, status="failed", message=str(e))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _parse_timedtext(text: str, fmt: str = "") -> list:
    """解析 YouTube timedtext（XML 或 json3）。"""
    import re as _re

    segments = []

    # JSON (json3)
    if text.strip().startswith("{"):
        try:
            data = json.loads(text)
            for ev in data.get("events", []):
                if not ev.get("segs"):
                    continue
                t = "".join(s.get("utf8", "") for s in ev["segs"]).strip()
                if t:
                    segments.append({
                        "start": round(ev.get("tStartMs", 0) / 1000, 2),
                        "dur": round(ev.get("dDurationMs", 0) / 1000, 2),
                        "text": t,
                    })
        except Exception:
            pass
        return segments

    # XML — srv3 用 <p>, srv1/預設用 <text>
    for tag in ["text", "p"]:
        for m in _re.finditer(rf'<{tag}\s+([^>]*)>([\s\S]*?)</{tag}>', text):
            attrs = m.group(1)
            raw = m.group(2)
            # srv3: t="33" d="2533"（毫秒）; srv1: start="0.5" dur="2.0"（秒）
            sm = _re.search(r'(?:t|start)="([\d.]+)"', attrs)
            dm = _re.search(r'(?:d|dur)="([\d.]+)"', attrs)
            start_raw = float(sm.group(1)) if sm else 0
            dur_raw = float(dm.group(1)) if dm else 0
            # srv3 的 t/d 是毫秒
            if tag == "p":
                start = round(start_raw / 1000, 2)
                dur = round(dur_raw / 1000, 2)
            else:
                start = round(start_raw, 2)
                dur = round(dur_raw, 2)
            # decode entities
            clean = (raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                       .replace("&#39;", "'").replace("&quot;", '"').replace("&apos;", "'").strip())
            if clean:
                segments.append({"start": start, "dur": dur, "text": clean})
        if segments:
            break

    return segments


def _moss_transcribe(audio_path: str, language: Optional[str] = None):
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"model": "moss", "response_format": "verbose_json"}
        if language:
            data["language"] = language
        resp = requests.post(MOSS_URL, files=files, data=data, timeout=600)

    if resp.status_code != 200:
        raise Exception(f"MOSS error {resp.status_code}: {resp.text[:200]}")

    result = resp.json()
    segments = []
    for seg in result.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", start)
        segments.append({
            "start": round(start, 2),
            "dur": round(end - start, 2),
            "text": seg.get("text", "").strip(),
        })
    return segments, result.get("language", "unknown")


def extract_video_id(url: str) -> Optional[str]:
    for p in [
        r"youtube\.com/watch\?v=([\w-]+)",
        r"youtu\.be/([\w-]+)",
        r"youtube\.com/shorts/([\w-]+)",
    ]:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting on port %d, MOSS at %s", PORT, MOSS_URL)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

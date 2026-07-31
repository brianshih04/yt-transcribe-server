#!/bin/bash
cd /home/avuser/yt-transcribe-server
./venv/bin/pip3 install uvicorn requests pydantic yt-dlp 2>&1
echo "EXIT: $?"

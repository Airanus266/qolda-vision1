@echo off
cd /d "%~dp0"
echo Open http://localhost:8000 in Chrome or Edge.
echo Keep this window open. Press Ctrl+C to stop.
py -3 -m http.server 8000 --directory dist
if errorlevel 1 pause

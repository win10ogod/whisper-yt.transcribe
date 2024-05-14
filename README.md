# 使用方法
# 警告安裝必要的庫：

pip install torch transformers yt-dlp librosa

# 運行轉錄工具：
從 YouTube 轉錄：

python whisper_transcribe.py --youtube YOUR_YOUTUBE_URL --output output.json

# 從本機音源文件或影片文件轉錄：

python whisper_transcribe.py --file YOUR_AUDIO_OR_VIDEO_FILE --output output.json

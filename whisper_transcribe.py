import os
import json
import torch
import yt_dlp
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# 設置模型
model_name = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# 使用 ASR Pipeline
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    return_timestamps=True
)

def transcribe_audio_segments(audio_path):
    """
    將音頻文件精確分段轉錄為文本
    """
    import librosa

    # 加載音頻文件
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # 使用 ASR Pipeline 進行轉錄
    result = asr_pipeline(audio, return_timestamps="word")
    
    # 整理段落
    segments = []
    current_segment = {"start": None, "end": None, "text": ""}
    for word in result["chunks"]:
        if current_segment["start"] is None:
            current_segment["start"] = word["timestamp"][0]
        current_segment["end"] = word["timestamp"][1]
        current_segment["text"] += word["text"] + " "
        
        if word["text"].endswith((".", "!", "?", "\n")):
            current_segment["text"] = current_segment["text"].strip()
            segments.append(current_segment)
            current_segment = {"start": None, "end": None, "text": ""}
    
    if current_segment["text"]:
        current_segment["text"] = current_segment["text"].strip()
        segments.append(current_segment)

    return segments

def save_transcriptions_to_json(transcriptions, output_path):
    """
    將轉錄的文本分段保存為 JSON 格式
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": transcriptions}, f, ensure_ascii=False, indent=4)

def download_and_transcribe_youtube(url, output_path):
    """
    下載 YouTube 視頻並將其分段轉錄為文本，最終保存為 JSON 格式
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    transcriptions = transcribe_audio_segments("temp_audio.mp3")
    save_transcriptions_to_json(transcriptions, output_path)
    os.remove("temp_audio.mp3")

def transcribe_local_file(file_path, output_path):
    """
    將本機音頻或視頻文件分段轉錄為文本，最終保存為 JSON 格式
    """
    transcriptions = transcribe_audio_segments(file_path)
    save_transcriptions_to_json(transcriptions, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper Transcription Tool with Accurate Segments")
    parser.add_argument("--youtube", type=str, help="YouTube video URL to transcribe")
    parser.add_argument("--file", type=str, help="Local audio/video file path to transcribe")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    if args.youtube:
        download_and_transcribe_youtube(args.youtube, args.output)
    elif args.file:
        transcribe_local_file(args.file, args.output)
    else:
        print("Please provide either a YouTube URL or a local file path.")
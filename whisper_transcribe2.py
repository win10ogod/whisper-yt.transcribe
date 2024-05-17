import os
import json
import torch
import yt_dlp
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import librosa
import pandas as pd

model_name = "openai/whisper-large-v3"

# Set up the model
device = "cuda"
torch_dtype = torch.bfloat16
processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(device)

# Set up the ASR pipeline
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    return_timestamps=True,
)

def transcribe_audio_segments(audio_path):
    # 加載音頻文件
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # 使用 ASR Pipeline 進行轉錄
    result = asr_pipeline(audio, return_timestamps='word')

    # 整理段落
    segments = []
    current_segment = {"start": None, "end": None, "text": ""}
    last_end_time = 0

    for word in result["chunks"]:
        start_time, end_time = word["timestamp"]

        # 檢查單詞之間的間隔，如果超過0.5秒，則視為新段落
        if start_time - last_end_time > 0.45 and current_segment["text"]:
            current_segment["text"] = current_segment["text"].strip()
            segments.append(current_segment)
            current_segment = {"start": start_time, "end": None, "text": ""}

        if current_segment["start"] is None:
            current_segment["start"] = start_time
        current_segment["end"] = end_time
        current_segment["text"] += word["text"] + " "
        last_end_time = end_time

    if current_segment["text"]:
        current_segment["text"] = current_segment["text"].strip()
        segments.append(current_segment)

    return segments

def save_transcriptions_to_txt(transcriptions, output_path, pure_text=False):
    """
    Save transcriptions to a TXT file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        if pure_text:
            for segment in transcriptions:
                f.write(f"{segment['text']}\n\n")
        else:
            for segment in transcriptions:
                start_time = f"{segment['start']:.2f}"
                end_time = f"{segment['end']:.2f}"
                text = segment['text']
                f.write(f"[{start_time} - {end_time}] {text}\n\n")

def save_transcriptions_to_json(transcriptions, output_path):
    """
    Save transcriptions to a JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": transcriptions}, f, ensure_ascii=False, indent=4)

def save_transcriptions_to_parquet(transcriptions, output_path):
    """
    Save transcriptions to a Parquet file
    """
    df = pd.DataFrame(transcriptions)
    df.to_parquet(output_path, index=False)

def download_and_transcribe_youtube(url, output_txt_path=None, output_json_path=None, output_parquet_path=None, pure_text=False):
    """
    Download YouTube video and transcribe it with accurate pause detection
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["ios"]
            }
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    transcriptions = transcribe_audio_segments("temp_audio.mp3")
    if output_txt_path:
        save_transcriptions_to_txt(transcriptions, output_txt_path, pure_text)
    if output_json_path:
        save_transcriptions_to_json(transcriptions, output_json_path)
    if output_parquet_path:
        save_transcriptions_to_parquet(transcriptions, output_parquet_path)
    os.remove("temp_audio.mp3")

def transcribe_local_file(file_path, output_txt_path=None, output_json_path=None, output_parquet_path=None, pure_text=False):
    """
    Transcribe local audio or video file with accurate pause detection
    """
    transcriptions = transcribe_audio_segments(file_path)
    if output_txt_path:
        save_transcriptions_to_txt(transcriptions, output_txt_path, pure_text)
    if output_json_path:
        save_transcriptions_to_json(transcriptions, output_json_path)
    if output_parquet_path:
        save_transcriptions_to_parquet(transcriptions, output_parquet_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper Transcription Tool with Accurate Segments")
    parser.add_argument("--youtube", type=str, help="YouTube video URL to transcribe")
    parser.add_argument("--file", type=str, help="Local audio/video file path to transcribe")
    parser.add_argument("--output-txt", type=str, help="Output TXT file path")
    parser.add_argument("--output-json", type=str, help="Output JSON file path")
    parser.add_argument("--output-parquet", type=str, help="Output Parquet file path")
    parser.add_argument("--pure-text", action="store_true", help="Save only pure text without timestamps")
    args = parser.parse_args()

    if args.youtube:
        download_and_transcribe_youtube(args.youtube, args.output_txt, args.output_json, args.output_parquet, args.pure_text)
    elif args.file:
        transcribe_local_file(args.file, args.output_txt, args.output_json, args.output_parquet, args.pure_text)
    else:
        print("Please provide either a YouTube URL or a local file path.")
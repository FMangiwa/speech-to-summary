import torch
import config
import tempfile
import warnings
from transformers import pipeline
from visualizer import TokenPredictor

# Suppress Hugging Face transformers deprecation & chunking warnings
warnings.filterwarnings("ignore", category=UserWarning)

_whisper_pipeline = None


def get_whisper_pipeline():
    global _whisper_pipeline
    if _whisper_pipeline is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=device,
            ignore_warning=True,
        )
    return _whisper_pipeline


def format_timestamp(seconds: float, srt_format: bool = True) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    separator = "," if srt_format else "."
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{separator}{millis:03d}"


def transcribe_audio(audio_path: str):
    pipe = get_whisper_pipeline()

    # Pass return_timestamps="word" so Whisper doesn't group 10 minutes into 1 chunk
    result = pipe(
        audio_path,
        return_timestamps="word",
    )

    chunks = result.get("chunks", [])

    # Group word-level chunks into ~10 second blocks for clean lines & SRT
    grouped_segments = []
    current_words = []
    current_start = None
    current_end = 0.0

    for chunk in chunks:
        text = chunk["text"]
        timestamp = chunk.get("timestamp", (0.0, 0.0))
        start = timestamp[0] if timestamp[0] is not None else current_end
        end = timestamp[1] if timestamp[1] is not None else start + 0.5

        if current_start is None:
            current_start = start

        current_words.append(text)
        current_end = end

        # Split into a new timestamp line every 10 seconds or on sentence punctuation (. ? !)
        if (end - current_start >= 10.0) or any(text.endswith(p) for p in [".", "?", "!"]):
            segment_text = "".join(current_words).strip()
            if segment_text:
                grouped_segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": segment_text
                })
            current_words = []
            current_start = None

    # Append any remaining words
    if current_words:
        segment_text = "".join(current_words).strip()
        if segment_text:
            grouped_segments.append({
                "start": current_start or 0.0,
                "end": current_end,
                "text": segment_text
            })

    # 1. Format Readable Transcript Output
    timestamped_lines = []
    for seg in grouped_segments:
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        timestamped_lines.append(f"[{mins:02d}:{secs:02d}] {seg['text']}")
    
    formatted_text = "\n".join(timestamped_lines)

    # 2. Build Line-by-Line SRT & VTT files
    srt_lines = []
    vtt_lines = ["WEBVTT\n"]

    for idx, seg in enumerate(grouped_segments, 1):
        srt_start = format_timestamp(seg["start"], srt_format=True)
        srt_end = format_timestamp(seg["end"], srt_format=True)
        srt_lines.append(f"{idx}\n{srt_start} --> {srt_end}\n{seg['text']}\n")

        vtt_start = format_timestamp(seg["start"], srt_format=False)
        vtt_end = format_timestamp(seg["end"], srt_format=False)
        vtt_lines.append(f"{vtt_start} --> {vtt_end}\n{seg['text']}\n")

    srt_content = "\n".join(srt_lines)
    vtt_content = "\n".join(vtt_lines)

    srt_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8")
    srt_temp.write(srt_content)
    srt_temp.close()

    vtt_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".vtt", mode="w", encoding="utf-8")
    vtt_temp.write(vtt_content)
    vtt_temp.close()

    return formatted_text, srt_content, vtt_content, srt_temp.name, vtt_temp.name


def generate_minutes(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript text available to summarize."

    # Strip out timestamps like [00:00] to evaluate actual word count
    clean_words = [w for w in transcript.split() if not (w.startswith("[") and w.endswith("]"))]
    
    # If the transcript is just a mic test or under 15 words, skip deep meeting summary
    if len(clean_words) < 15:
        return (
            "### 📌 Audio Test / Short Clip Detected\n"
            f"**Transcript:** \"{' '.join(clean_words)}\"\n\n"
            "*(Note: The audio recording is too short to generate full meeting minutes.)*"
        )

    # Standard LLM prompt for longer recordings
    prompt = (
        "You are an expert executive assistant. Provide a concise summary of the following meeting transcript. "
        "Do NOT invent or extrapolate details that are not directly stated in the text.\n\n"
        f"Transcript:\n{transcript}"
    )

    predictor = TokenPredictor(model_name=config.OPENAI_MODEL)
    summary_text, predictions = predictor.predict_tokens(prompt, max_tokens=1200)

    return summary_text

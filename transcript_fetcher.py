import re
from typing import List, Dict, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


def extract_video_id(url: str) -> str:
   
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def seconds_to_label(seconds: float) -> str:
    
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fetch_transcript(video_id: str, chunk_duration_seconds: int = 60) -> Tuple[List[Dict], Dict]:

    try:
        # Prefer manual captions; fall back to auto-generated
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            transcript = transcript_list.find_generated_transcript(["en"])

        raw_entries = [{"text": s.text, "start": s.start, "duration": s.duration}
                       for s in transcript.fetch()]

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No English transcript found. Try a video with English captions.")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {str(e)}")

    # Chunk transcript by time window 
    chunks = []
    current_text = []
    chunk_start = raw_entries[0]["start"]

    for entry in raw_entries:
        current_text.append(entry["text"].strip())
        entry_end = entry["start"] + entry.get("duration", 0)

        if entry_end - chunk_start >= chunk_duration_seconds:
            chunk_text = " ".join(current_text)
            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": entry_end,
                "label": seconds_to_label(chunk_start),
                "start_seconds": int(chunk_start),
            })
            current_text = []
            chunk_start = entry_end

    # Add remaining text
    if current_text:
        last_end = raw_entries[-1]["start"] + raw_entries[-1].get("duration", 0)
        chunks.append({
            "text": " ".join(current_text),
            "start": chunk_start,
            "end": last_end,
            "label": seconds_to_label(chunk_start),
            "start_seconds": int(chunk_start),
        })

    metadata = {
        "video_id": video_id,
        "total_chunks": len(chunks),
        "total_duration_seconds": chunks[-1]["end"] if chunks else 0,
    }

    return chunks, metadata
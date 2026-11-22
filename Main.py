
import os
import json
import uuid
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_UPLOAD = 20 * 1024 * 1024  # 20 MB demo limit
OUTPUT_DIR = Path("/tmp/clip_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Clip Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://devikharel4567-creator.github.io"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def timestamp(seconds):
    ms = int(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def create_srt(segments, start, end):
    lines = []
    index = 1

    for segment in segments:
        s = float(segment["start"])
        e = float(segment["end"])

        if e <= start or s >= end:
            continue

        s = max(s, start) - start
        e = min(e, end) - start

        lines.extend([
            str(index),
            f"{timestamp(s)} --> {timestamp(e)}",
            segment["text"].strip(),
            ""
        ])
        index += 1

    return "\n".join(lines)


@app.get("/")
def home():
    return {
        "name": "AI Clip Studio 4.0",
        "status": "Backend online",
        "features": [
            "Video upload",
            "AI transcription",
            "AI highlights",
            "Automatic clips",
            "Subtitles"
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/process")
async def process_video(
    request_file: UploadFile = File(...),
    clip_count: int = Form(3),
    language: str = Form("")
):
    if not client:
        raise HTTPException(500, "GROQ_API_KEY is missing")

    allowed = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    suffix = Path(request_file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, "Upload an MP4, MOV, MKV or WebM video")

    if not 1 <= clip_count <= 5:
        raise HTTPException(400, "Choose between 1 and 5 clips")

    job_id = uuid.uuid4().hex

    try:
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            video = temp / f"video{suffix}"
            audio = temp / "audio.mp3"

            # Save uploaded video with a size limit
            size = 0
            with video.open("wb") as f:
                while True:
                    chunk = await request_file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(
                            413, "Video is too large. Demo limit is 20 MB."
                        )
                    f.write(chunk)

            # Read video duration
            info = json.loads(run_command([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(video)
            ]))

            duration = float(info["format"]["duration"])

            if duration <= 0 or duration > 1800:
                raise HTTPException(
                    400, "Video must be between 1 second and 30 minutes"
                )

            # Extract and compress audio
            run_command([
                "ffmpeg", "-y", "-i", str(video),
                "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", "48k", str(audio)
            ])

            if audio.stat().st_size > 24 * 1024 * 1024:
                raise HTTPException(
                    413, "Audio is too large for the transcription service"
                )

            # AI transcription
            with audio.open("rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=(audio.name, audio_file),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=language or None,
                    temperature=0
                )

            segments = [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text
                }
                for s in transcript.segments
            ]

            if not segments:
                raise HTTPException(422, "No speech was detected")

            # AI highlight detection
            prompt = f"""
Analyze this video transcript and select {clip_count}
interesting short-form video moments.

Choose moments with strong hooks, useful information,
surprising statements, humor or engaging conversation.

Return ONLY a JSON object:
{{
  "clips": [
    {{
      "start": 10,
      "end": 35,
      "title": "Short engaging title",
      "reason": "Why this moment is interesting"
    }}
  ]
}}

Rules:
- All timestamps must be in seconds.
- Start and end must be within the video duration.
- Each clip must be at least 10 seconds if possible.
- Avoid repetitive clips.
- Do not invent dialogue.
- Video duration: {duration} seconds.

TRANSCRIPT:
{json.dumps(segments, ensure_ascii=False)}
"""

            result = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an accurate video highlight editor. Return valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            ai_data = json.loads(
                result.choices[0].message.content
            )

            clips = ai_data.get("clips", [])[:clip_count]
            results = []

            # Generate clips and subtitle files
            for i, clip in enumerate(clips, 1):
                start = max(0, float(clip["start"]))
                end = min(duration, float(clip["end"]))

                if end - start < 3:
                    continue

                clip_id = f"{job_id}_{i}"
                mp4 = OUTPUT_DIR / f"{clip_id}.mp4"
                srt = OUTPUT_DIR / f"{clip_id}.srt"

                srt.write_text(
                    create_srt(segments, start, end),
                    encoding="utf-8"
                )

                run_command([
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video),
                    "-t", str(end - start),
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(mp4)
                ])

                results.append({
                    "title": str(clip.get("title", f"Clip {i}")),
                    "reason": str(clip.get("reason", "")),
                    "start": start,
                    "end": end,
                    "video_url": f"/outputs/{clip_id}.mp4",
                    "subtitle_url": f"/outputs/{clip_id}.srt"
                })

            if not results:
                raise HTTPException(422, "No valid clips were generated")

            return {
                "success": True,
                "message": "Video processed successfully",
                "transcript": transcript.text,
                "clips": results
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)[:500]}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

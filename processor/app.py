import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="ABR Video Transcoding Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "dash_output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/video", StaticFiles(directory=str(OUTPUT_DIR)), name="video")

# ffmpeg preset 越快，转码越快但压缩效率越差。毕设演示用 medium/veryfast 都可以。
FFMPEG_PRESET = "medium"
FPS_ASSUMPTION = 30
SEGMENT_DURATION = 4
GOP_SIZE = FPS_ASSUMPTION * SEGMENT_DURATION

# 与 agent/settings.py 的 BITRATES_KBPS 保持一致：300/600/900/2000/4500/9000 Kbps。
DASH_TIERS = [
    {
        "name": "1440p",
        "height": 1440,
        "resolution": "2560x1440",
        "bitrate": "9000k",
        "maxrate": "9900k",
        "bufsize": "18000k",
        "profile": "high",
    },
    {
        "name": "1080p",
        "height": 1080,
        "resolution": "1920x1080",
        "bitrate": "4500k",
        "maxrate": "4950k",
        "bufsize": "9000k",
        "profile": "high",
    },
    {
        "name": "720p",
        "height": 720,
        "resolution": "1280x720",
        "bitrate": "2000k",
        "maxrate": "2200k",
        "bufsize": "4000k",
        "profile": "high",
    },
    {
        "name": "480p",
        "height": 480,
        "resolution": "854x480",
        "bitrate": "900k",
        "maxrate": "1000k",
        "bufsize": "1800k",
        "profile": "main",
    },
    {
        "name": "360p",
        "height": 360,
        "resolution": "640x360",
        "bitrate": "600k",
        "maxrate": "660k",
        "bufsize": "1200k",
        "profile": "main",
    },
    {
        "name": "240p",
        "height": 240,
        "resolution": "426x240",
        "bitrate": "300k",
        "maxrate": "330k",
        "bufsize": "600k",
        "profile": "baseline",
    },
]


def get_video_height(filepath: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(filepath),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception as exc:
        print(f"[Processor] failed to read video height, fallback to 1080p: {exc}")
        return 1080


def build_ffmpeg_command(input_filepath: Path, video_id: str, target_tiers):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_filepath),
        "-c:v",
        "libx264",
        "-preset",
        FFMPEG_PRESET,
        "-g",
        str(GOP_SIZE),
        "-keyint_min",
        str(GOP_SIZE),
        "-sc_threshold",
        "0",
    ]

    for idx, tier in enumerate(target_tiers):
        command.extend(
            [
                "-map",
                "0:v:0",
                f"-b:v:{idx}",
                tier["bitrate"],
                f"-maxrate:v:{idx}",
                tier["maxrate"],
                f"-bufsize:v:{idx}",
                tier["bufsize"],
                f"-s:v:{idx}",
                tier["resolution"],
                f"-profile:v:{idx}",
                tier["profile"],
            ]
        )

    command.extend(["-map", "0:a:0?", "-c:a", "aac", "-b:a:0", "128k"])
    command.extend(
        [
            "-use_timeline",
            "1",
            "-use_template",
            "1",
            "-window_size",
            "0",
            "-seg_duration",
            str(SEGMENT_DURATION),
            "-init_seg_name",
            "init_$RepresentationID$.m4s",
            "-media_seg_name",
            "chunk_$RepresentationID$_$Number%05d$.m4s",
            "-adaptation_sets",
            "id=0,streams=v id=1,streams=a",
            "-f",
            "dash",
            f"{video_id}.mpd",
        ]
    )
    return command


def process_video_to_dash(input_filepath: str, video_id: str):
    input_path = Path(input_filepath)
    output_folder = OUTPUT_DIR / video_id
    output_folder.mkdir(parents=True, exist_ok=True)

    source_height = get_video_height(input_path)
    target_tiers = [
        tier for tier in DASH_TIERS if tier["height"] <= source_height + 10
    ] or [DASH_TIERS[-1]]
    command = build_ffmpeg_command(input_path, video_id, target_tiers)

    try:
        print(
            f"[Processor] {video_id}: transcoding {len(target_tiers)} video tiers "
            f"from source_height={source_height}p"
        )
        subprocess.run(
            command,
            check=True,
            cwd=output_folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[Processor] {video_id}: DASH output saved to {output_folder}")
    except subprocess.CalledProcessError as exc:
        print(f"[Processor] {video_id}: ffmpeg failed\n{exc.stderr}")


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".mp4"):
        return JSONResponse(
            status_code=400,
            content={"message": "目前仅支持 MP4 格式"},
        )

    video_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{video_id}.mp4"

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_video_to_dash, str(save_path), video_id)

    return {
        "status": "success",
        "message": "视频已接收，正在后台转码。",
        "video_id": video_id,
        "play_url": f"/video/{video_id}/{video_id}.mpd",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)

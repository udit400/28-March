import requests
from typing import Optional

TRUTHSCAN_IMAGE_URL = "https://truthscan.com/api/ai-image-detector"
TRUTHSCAN_VIDEO_URL = "https://truthscan.com/api/deepfake-detector"
TRUTHSCAN_VOICE_URL = "https://truthscan.com/api/ai-voice-detector"


def analyze_image_with_truthscan(image_bytes: bytes, filename: str = "image.jpg") -> Optional[dict]:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    try:
        response = requests.post(TRUTHSCAN_IMAGE_URL, files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_video_with_truthscan(video_bytes: bytes, filename: str = "video.mp4") -> Optional[dict]:
    files = {"file": (filename, video_bytes, "video/mp4")}
    try:
        response = requests.post(TRUTHSCAN_VIDEO_URL, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_voice_with_truthscan(audio_bytes: bytes, filename: str = "audio.wav") -> Optional[dict]:
    files = {"file": (filename, audio_bytes, "audio/wav")}
    try:
        response = requests.post(TRUTHSCAN_VOICE_URL, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

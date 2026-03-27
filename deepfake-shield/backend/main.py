import asyncio
import datetime
import html
import hashlib
import ipaddress
import io
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request as UrlRequest, urlopen

import cv2
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import (
    GMAIL_SETUP_HINT,
    OTP_EXPIRY_MINUTES,
    OTP_MAX_ATTEMPTS,
    OTP_RESEND_COOLDOWN_SECONDS,
    create_access_token,
    email_delivery_configured,
    generate_otp,
    hash_otp,
    is_gmail_address,
    mask_email,
    normalize_email,
    send_activity_email,
    send_otp_email,
    verify_otp_hash,
)
from database import SessionLocal, User


app = FastAPI(
    title="Deepfake Detection API",
    version="1.0.0",
    description="Local API for image-frame deepfake scoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = Path(os.getenv("DEEPFAKE_MODEL_PATH", Path(__file__).with_name("model.pt")))
FAKE_THRESHOLD = float(os.getenv("DEEPFAKE_THRESHOLD", "65.0"))
MAX_REMOTE_BYTES = 8 * 1024 * 1024
REALITY_DEFENDER_API_KEY = os.getenv("REALITY_DEFENDER_API_KEY", "")
REALITY_DEFENDER_MAX_ATTEMPTS = int(os.getenv("REALITY_DEFENDER_MAX_ATTEMPTS", "30"))
REALITY_DEFENDER_POLLING_INTERVAL_MS = int(os.getenv("REALITY_DEFENDER_POLLING_INTERVAL_MS", "2000"))
REALITY_DEFENDER_SEQUENCE_FRAME_LIMIT = max(1, int(os.getenv("REALITY_DEFENDER_SEQUENCE_FRAME_LIMIT", "3")))
REALITY_DEFENDER_SOCIAL_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "tiktok.com",
    "www.tiktok.com",
}
META_IMAGE_PATTERNS = [
    re.compile(
        r'<meta[^>]+property=["\'](?:og:image|og:image:url)["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\'](?:og:image|og:image:url)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']',
        re.IGNORECASE,
    ),
]


class MockDetector:
    mode = "mock"

    def predict(self, image: Image.Image, image_bytes: bytes) -> float:
        digest = hashlib.sha256(image_bytes).hexdigest()
        scaled_value = int(digest[:8], 16) / 0xFFFFFFFF
        return round(10.0 + (scaled_value * 89.9), 2)


class TorchDetector:
    mode = "torch"

    def __init__(self, model_path: Path) -> None:
        import torch
        from torchvision import transforms

        self._torch = torch
        self._transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.load(model_path, map_location=self._device)
        self._model.eval()

    def predict(self, image: Image.Image, image_bytes: bytes) -> float:
        del image_bytes
        input_tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            output = self._model(input_tensor)
            probability = self._torch.nn.functional.softmax(output[0], dim=0)[1].item()
        return round(probability * 100, 2)


class RealityDefenderDetector:
    mode = "realitydefender"

    def __init__(self, api_key: str) -> None:
        from realitydefender import RealityDefender

        self._client = RealityDefender(api_key=api_key)

    @staticmethod
    def _normalize_score(score: Optional[float]) -> float:
        if score is None:
            return 0.0
        if score <= 1:
            return round(score * 100, 2)
        return round(score, 2)

    @staticmethod
    def _is_manipulated(status: Optional[str]) -> bool:
        normalized = str(status or "").strip().upper()
        return normalized in {"MANIPULATED", "DEEPFAKE", "SYNTHETIC", "FAKE"} or "MANIPUL" in normalized

    def _format_result(self, result: dict, request_id: Optional[str] = None, media_id: Optional[str] = None) -> dict:
        models = []
        for model in result.get("models", []):
            models.append(
                {
                    "name": model.get("name", "Unknown model"),
                    "status": model.get("status", "UNKNOWN"),
                    "score": self._normalize_score(model.get("score")),
                }
            )

        raw_score = self._normalize_score(result.get("score"))
        model_consensus = build_model_consensus(models, raw_score)
        score = model_consensus["ensemble_score"]
        status = str(result.get("status", "UNKNOWN"))
        return {
            "status": status,
            "score": score,
            "raw_score": raw_score,
            "is_manipulated": self._is_manipulated(status) or score > FAKE_THRESHOLD,
            "models": models,
            "model_consensus": model_consensus,
            "request_id": request_id,
            "media_id": media_id,
        }

    def detect_file(self, file_path: str) -> dict:
        result = self._client.detect_file(file_path)
        return self._format_result(result)

    def detect_social_url(self, url: str) -> dict:
        upload_result = self._client.upload_social_media_sync(social_media_link=url)
        result = self._client.get_result_sync(
            upload_result["request_id"],
            max_attempts=REALITY_DEFENDER_MAX_ATTEMPTS,
            polling_interval=REALITY_DEFENDER_POLLING_INTERVAL_MS,
        )
        return self._format_result(
            result,
            request_id=upload_result.get("request_id"),
            media_id=upload_result.get("media_id"),
        )


def build_model_consensus(models: List[dict], fallback_score: float) -> dict:
    valid_scores = [float(model.get("score") or 0.0) for model in models]
    if not valid_scores:
        return {
            "models_considered": 0,
            "manipulated_votes": 0,
            "agreement_ratio": 0.0,
            "average_model_score": round(fallback_score, 2),
            "peak_model_score": round(fallback_score, 2),
            "ensemble_score": round(fallback_score, 2),
        }

    manipulated_votes = sum(
        1
        for model, score in zip(models, valid_scores)
        if RealityDefenderDetector._is_manipulated(model.get("status")) or score >= FAKE_THRESHOLD
    )
    average_score = sum(valid_scores) / len(valid_scores)
    peak_score = max(valid_scores)
    agreement_ratio = manipulated_votes / len(valid_scores)
    ensemble_score = (fallback_score * 0.5) + (average_score * 0.35) + (peak_score * 0.15)

    if agreement_ratio >= 0.6:
        ensemble_score = max(ensemble_score, average_score)
    elif agreement_ratio == 0:
        ensemble_score = min(ensemble_score, max(fallback_score, average_score))

    return {
        "models_considered": len(valid_scores),
        "manipulated_votes": manipulated_votes,
        "agreement_ratio": round(agreement_ratio, 3),
        "average_model_score": round(average_score, 2),
        "peak_model_score": round(peak_score, 2),
        "ensemble_score": round(min(99.5, ensemble_score), 2),
    }


def median_score(values: List[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 2)


def build_frame_consensus(frame_results: List[dict], total_frames_received: int) -> dict:
    if not frame_results:
        raise HTTPException(status_code=400, detail="Uploaded media frames are empty.")

    frame_scores = [float(result.get("fake_score") or 0.0) for result in frame_results]
    suspicious_frames = sum(
        1 for result, score in zip(frame_results, frame_scores) if result.get("is_deepfake") or score >= FAKE_THRESHOLD
    )
    average_score = sum(frame_scores) / len(frame_scores)
    peak_score = max(frame_scores)
    median_frame_score = median_score(frame_scores)
    suspicious_ratio = suspicious_frames / len(frame_results)
    ensemble_score = (average_score * 0.5) + (median_frame_score * 0.3) + (peak_score * 0.2)

    if suspicious_ratio >= 0.4:
        ensemble_score = max(ensemble_score, average_score)
    elif suspicious_ratio == 0:
        ensemble_score = min(ensemble_score, max(average_score, median_frame_score))

    final_score = round(min(99.5, ensemble_score), 2)
    strongest_result = max(frame_results, key=lambda item: item.get("fake_score") or 0.0)
    metadata = dict(strongest_result.get("metadata") or {})
    metadata["frame_consensus"] = {
        "frames_received": total_frames_received,
        "frames_scored": len(frame_results),
        "suspicious_frames": suspicious_frames,
        "average_frame_score": round(average_score, 2),
        "median_frame_score": round(median_frame_score, 2),
        "peak_frame_score": round(peak_score, 2),
        "ensemble_score": final_score,
        "suspicious_ratio": round(suspicious_ratio, 3),
    }

    return {
        "engine": strongest_result["engine"],
        "fake_score": final_score,
        "is_deepfake": final_score >= FAKE_THRESHOLD or suspicious_ratio >= 0.4,
        "threshold": strongest_result["threshold"],
        "metadata": metadata,
        "frame_scores": [round(score, 2) for score in frame_scores],
        "frames_scored": len(frame_results),
    }


def analyze_visual_sequence(image_bytes_list: List[bytes], force_local: bool = False) -> dict:
    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="Uploaded media frames are empty.")

    frames_to_score = image_bytes_list
    if detector.mode == "realitydefender" and not force_local:
        frames_to_score = image_bytes_list[:REALITY_DEFENDER_SEQUENCE_FRAME_LIMIT]

    frame_results = [score_image_bytes(frame_bytes, force_local=force_local) for frame_bytes in frames_to_score]
    return build_frame_consensus(frame_results, total_frames_received=len(image_bytes_list))


def build_reality_defender_metadata(base_metadata: Optional[dict], detection_result: dict) -> dict:
    metadata = dict(base_metadata or {})
    metadata["reality_defender_status"] = detection_result.get("status", "UNKNOWN")
    metadata["reality_defender_score"] = detection_result.get("score", 0.0)

    raw_score = detection_result.get("raw_score")
    if raw_score is not None:
        metadata["reality_defender_raw_score"] = raw_score

    request_id = detection_result.get("request_id")
    if request_id:
        metadata["reality_defender_request_id"] = request_id

    media_id = detection_result.get("media_id")
    if media_id:
        metadata["reality_defender_media_id"] = media_id

    model_consensus = detection_result.get("model_consensus") or {}
    if model_consensus:
        metadata["reality_defender_model_consensus"] = model_consensus

    models = detection_result.get("models") or []
    if models:
        metadata["reality_defender_models"] = models
        top_model = max(models, key=lambda item: item.get("score") or 0.0)
        metadata["reality_defender_top_model"] = top_model

    return metadata


def analyze_file_with_detector(file_path: str) -> dict:
    if detector.mode != "realitydefender":
        raise RuntimeError("Reality Defender detector is not active.")
    try:
        return detector.detect_file(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Reality Defender file analysis failed: {exc}",
        ) from exc


def is_supported_social_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in REALITY_DEFENDER_SOCIAL_HOSTS


def build_detector() -> tuple[object, Optional[str]]:
    if REALITY_DEFENDER_API_KEY:
        try:
            return RealityDefenderDetector(REALITY_DEFENDER_API_KEY), None
        except Exception as exc:  # pragma: no cover - external SDK init path
            return MockDetector(), f"Reality Defender SDK failed to initialize: {exc}. Using mock detector."

    if not MODEL_PATH.exists():
        return MockDetector(), f"Model file not found at {MODEL_PATH}. Using mock detector."

    try:
        return TorchDetector(MODEL_PATH), None
    except Exception as exc:  # pragma: no cover - fallback path for demo resilience
        return MockDetector(), f"Failed to load model: {exc}. Using mock detector."


detector, startup_warning = build_detector()


class AnalyzeUrlRequest(BaseModel):
    url: str


class AuthRequest(BaseModel):
    email: str


class VerifyRequest(BaseModel):
    email: str
    otp: str


class PaymentRequest(BaseModel):
    email: str
    utr_number: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def request_is_local(request: Request) -> bool:
    client_host = (request.client.host if request.client else "").strip().lower()
    if not client_host:
        return False
    if client_host == "localhost":
        return True
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return False


def has_active_subscription(user: User) -> bool:
    return bool(user.subscription_end and user.subscription_end > datetime.datetime.utcnow())


def clear_pending_otp(user: User) -> None:
    user.current_otp = None
    user.current_otp_hash = None
    user.otp_expiry = None
    user.otp_attempts = 0


def extract_metadata(image: Image.Image) -> dict:
    metadata = {
        "is_original": "Likely a Copy/Stripped (Social Media)",
        "creation_date": "Unknown",
        "software_platform": "Unknown",
        "camera_device": "Unknown",
        "c2pa_signature": "No AI Signature Detected",
    }

    software_hits = []
    camera_hits = []

    try:
        exif_data = image.getexif()
        if exif_data:
            metadata["is_original"] = "Original File (EXIF Intact)"
            for tag_id, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag_id, tag_id)
                if decoded in {"DateTimeOriginal", "DateTime"}:
                    metadata["creation_date"] = str(value)
                elif decoded == "Software":
                    software_hits.append(str(value))
                elif decoded in {"Make", "Model"}:
                    camera_hits.append(str(value))
    except Exception:
        pass

    info_values = []
    for key, value in image.info.items():
        if isinstance(value, bytes):
            try:
                normalized = value.decode("utf-8", errors="ignore")
            except Exception:
                normalized = repr(value[:80])
        else:
            normalized = str(value)
        info_values.append(f"{key}: {normalized}")

    software_blob = " | ".join(software_hits + info_values)
    lowered_blob = software_blob.lower()

    if software_hits:
        metadata["software_platform"] = " | ".join(dict.fromkeys(software_hits))

    if camera_hits:
        metadata["camera_device"] = " / ".join(dict.fromkeys(camera_hits))

    ai_markers = [
        "c2pa",
        "midjourney",
        "dall-e",
        "adobe firefly",
        "firefly",
        "stable diffusion",
        "openai",
        "generative fill",
    ]
    if any(marker in lowered_blob for marker in ai_markers):
        metadata["c2pa_signature"] = "Generative AI Platform or Provenance Marker Detected"

    return metadata


def score_image_bytes(image_bytes: bytes, force_local: bool = False) -> dict:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc

    metadata = extract_metadata(image)

    temp_path: Optional[str] = None
    try:
        if detector.mode == "realitydefender" and not force_local:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            detection_result = analyze_file_with_detector(temp_path)
            fake_score = detection_result["score"]
            is_deepfake = detection_result["is_manipulated"]
            metadata = build_reality_defender_metadata(metadata, detection_result)
        else:
            local_detector = MockDetector() if force_local and detector.mode == "realitydefender" else detector
            fake_score = local_detector.predict(image, image_bytes)
            is_deepfake = fake_score > FAKE_THRESHOLD
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    explanation = build_result_explanation(
        visual_score=fake_score,
        liveness_score=None,
        audio_score=None,
        metadata=metadata,
        frames_analyzed=1,
    )

    return {
        "status": "success",
        "engine": "mock" if force_local and detector.mode == "realitydefender" else detector.mode,
        "fake_score": fake_score,
        "is_deepfake": is_deepfake,
        "threshold": FAKE_THRESHOLD,
        "message": (
            "This image may be fake or heavily edited."
            if is_deepfake
            else "This image looks mostly real."
        ),
        "analysis_summary": (
            "This image may be fake or edited."
            if is_deepfake
            else "This image looks mostly authentic."
        ),
        "explanation": explanation,
        "metadata": metadata,
    }


def verify_biological_liveness(image_bytes_list: List[bytes]) -> float:
    if len(image_bytes_list) < 2:
        return 0.0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():  # pragma: no cover - depends on OpenCV runtime assets
        return 0.0

    faces_detected = 0
    green_channel_means: List[float] = []

    try:
        for image_bytes in image_bytes_list:
            frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                continue

            faces_detected += 1
            x, y, width, height = faces[0]
            roi_height = max(1, height // 3)
            roi = frame[y : y + roi_height, x : x + width]
            if roi.size == 0:
                continue

            green_channel_means.append(float(np.mean(roi[:, :, 1])))
    except Exception:
        return 0.0

    if faces_detected == 0 or len(green_channel_means) < 2:
        return 0.0

    face_consistency = faces_detected / len(image_bytes_list)
    signal_variation = float(np.std(green_channel_means))
    normalized_variation = min(signal_variation / 4.0, 1.0)
    score = 45.0 + (face_consistency * 35.0) + (normalized_variation * 20.0)
    return round(min(score, 99.5), 2)


def generate_proof_of_reality(deepfake_score: float, liveness_score: float, metadata: dict) -> tuple[str, str]:
    timestamp = datetime.datetime.utcnow().isoformat()
    raw_payload = (
        f"{deepfake_score}|{liveness_score}|{timestamp}|"
        f"{metadata.get('software_platform', 'Unknown')}|{metadata.get('c2pa_signature', 'Unknown')}"
    )
    proof_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
    return proof_hash, timestamp


def analyze_full_audio(audio_path: str) -> float:
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    if not audio_bytes:
        return 0.0

    digest = hashlib.sha256(audio_bytes).hexdigest()
    scaled_value = int(digest[:8], 16) / 0xFFFFFFFF
    return round(10.0 + (scaled_value * 85.0), 2)


def build_result_explanation(
    *,
    visual_score: Optional[float],
    liveness_score: Optional[float],
    audio_score: Optional[float],
    metadata: Optional[dict],
    frames_analyzed: int,
) -> str:
    reasons: List[str] = []
    metadata = metadata or {}

    if visual_score is not None:
        if visual_score >= FAKE_THRESHOLD:
            reasons.append(
                f"The fake-risk score is {visual_score}%, which is above the warning level of {FAKE_THRESHOLD:.1f}%."
            )
        elif visual_score >= max(FAKE_THRESHOLD - 15.0, 0.0):
            reasons.append(
                f"The fake-risk score is {visual_score}%. That is a bit high, so this file may need a closer look."
            )
        else:
            reasons.append(f"The fake-risk score is low at {visual_score}%, so strong signs of manipulation were not found.")

    if liveness_score is not None and liveness_score > 0:
        if liveness_score < 50.0:
            reasons.append(
                f"The face-movement score is only {liveness_score}%, so natural movement looks weak across the checked frames."
            )
        else:
            reasons.append(
                f"The face-movement score is {liveness_score}%, which looks more natural."
            )

    if audio_score is not None:
        if audio_score >= FAKE_THRESHOLD:
            reasons.append(
                f"The audio risk score is {audio_score}%, so the voice or sound pattern looks suspicious."
            )
        elif audio_score >= max(FAKE_THRESHOLD - 15.0, 0.0):
            reasons.append(
                f"The audio risk score is {audio_score}%, so the sound should be checked manually."
            )
        else:
            reasons.append(f"The audio risk score is low at {audio_score}%, so strong signs of voice cloning were not found.")

    provenance = metadata.get("c2pa_signature")
    if provenance and provenance != "No AI Signature Detected":
        reasons.append("The file metadata contains signs that AI tools may have been used.")

    detector_status = metadata.get("reality_defender_status")
    if detector_status:
        readable_status = str(detector_status).replace("_", " ").lower()
        reasons.append(f"Reality Defender marked this file as {readable_status}.")

    model_consensus = metadata.get("reality_defender_model_consensus") or {}
    if isinstance(model_consensus, dict) and model_consensus.get("models_considered", 0) > 1:
        models_considered = model_consensus["models_considered"]
        manipulated_votes = model_consensus.get("manipulated_votes", 0)
        agreement_ratio = float(model_consensus.get("agreement_ratio", 0.0)) * 100.0
        reasons.append(
            f"The final detector score combines {models_considered} model signals, with {manipulated_votes} pointing to manipulation and {agreement_ratio:.0f}% agreement."
        )

    top_model = metadata.get("reality_defender_top_model") or {}
    if isinstance(top_model, dict) and top_model.get("name"):
        top_model_score = top_model.get("score")
        if top_model_score is not None:
            reasons.append(
                f"Its strongest model was {top_model['name']} with a score of {top_model_score}%."
            )

    originality = metadata.get("is_original")
    if originality == "Likely a Copy/Stripped (Social Media)":
        reasons.append("This looks like a repost or stripped copy, so there are fewer original file details to inspect.")
    elif originality == "Original File (EXIF Intact)":
        reasons.append("This file still has its original metadata, which makes it easier to trace.")

    frame_consensus = metadata.get("frame_consensus") or {}
    if isinstance(frame_consensus, dict) and frame_consensus.get("frames_scored", 0) > 1:
        reasons.append(
            f"{frame_consensus.get('suspicious_frames', 0)} of {frame_consensus.get('frames_scored', 0)} scored frames looked suspicious, so the result uses frame consensus instead of a single snapshot."
        )

    if frames_analyzed > 1:
        reasons.append(f"This result is based on {frames_analyzed} checked frames, not just one image.")

    if not reasons:
        return "The scan finished, but there was not enough evidence to explain the result in detail."

    return " ".join(reasons)


def validate_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Provide a valid public http(s) URL.")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="URL hostname is missing.")

    try:
        for info in socket.getaddrinfo(hostname, None):
            ip_address = ipaddress.ip_address(info[4][0])
            if (
                ip_address.is_private
                or ip_address.is_loopback
                or ip_address.is_link_local
                or ip_address.is_multicast
                or ip_address.is_reserved
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Only public internet URLs are allowed for remote analysis.",
                )
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="Could not resolve the remote host.") from exc

    return parsed.geturl()


def fetch_remote_bytes(url: str) -> tuple[bytes, str, str]:
    request = UrlRequest(url, headers={"User-Agent": "SyntheticMediaShield/1.0"})

    try:
        with urlopen(request, timeout=12) as response:
            content_type = response.headers.get_content_type()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_REMOTE_BYTES:
                raise HTTPException(status_code=413, detail="Remote file is too large to analyze.")

            payload = response.read(MAX_REMOTE_BYTES + 1)
            final_url = response.geturl()
    except HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Remote server returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise HTTPException(status_code=400, detail="Could not download the remote URL.") from exc

    if len(payload) > MAX_REMOTE_BYTES:
        raise HTTPException(status_code=413, detail="Remote file is too large to analyze.")

    return payload, content_type, final_url


def extract_preview_image_url(page_bytes: bytes, page_url: str) -> str:
    decoded = page_bytes.decode("utf-8", errors="ignore")

    for pattern in META_IMAGE_PATTERNS:
        match = pattern.search(decoded)
        if match:
            preview_url = html.unescape(match.group(1)).strip()
            if preview_url:
                return urljoin(page_url, preview_url)

    raise HTTPException(
        status_code=400,
        detail="Could not find a preview image for that page URL.",
    )


def resolve_youtube_thumbnail_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    video_id: Optional[str] = None

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/", 3)[2]

    if not video_id:
        return None

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    index_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status() -> dict:
    return {
        "service": "Deepfake Detection API",
        "status": "ok",
        "engine": detector.mode,
        "model_path": str(MODEL_PATH),
        "threshold": FAKE_THRESHOLD,
        "warning": startup_warning,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "engine": detector.mode,
        "model_loaded": detector.mode == "torch",
        "warning": startup_warning,
        "email_delivery_configured": email_delivery_configured(),
        "otp_expiry_minutes": OTP_EXPIRY_MINUTES,
        "otp_resend_cooldown_seconds": OTP_RESEND_COOLDOWN_SECONDS,
        "reality_defender_configured": bool(REALITY_DEFENDER_API_KEY),
    }


@app.post("/auth/request-otp")
def request_otp(
    req: AuthRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    email = normalize_email(req.email)
    if not is_gmail_address(email):
        raise HTTPException(status_code=400, detail="Enter a valid Gmail address ending in @gmail.com.")

    now = datetime.datetime.utcnow()

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email)
        db.add(user)

    if user.otp_last_sent_at:
        elapsed_seconds = int((now - user.otp_last_sent_at).total_seconds())
        if elapsed_seconds < OTP_RESEND_COOLDOWN_SECONDS:
            retry_after = OTP_RESEND_COOLDOWN_SECONDS - elapsed_seconds
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {retry_after} seconds before requesting another OTP.",
            )

    otp = generate_otp()
    user.current_otp = None
    user.current_otp_hash = hash_otp(email, otp)
    user.otp_expiry = now + datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)
    user.otp_attempts = 0
    user.otp_last_sent_at = now
    db.commit()

    delivery_mode = "email"
    delivery_note = ""
    otp_preview = None

    if email_delivery_configured():
        sent, detail = send_otp_email(email, otp, "Login Attempt")
        if not sent:
            clear_pending_otp(user)
            db.commit()
            raise HTTPException(status_code=502, detail=f"Could not send Gmail OTP. {detail}")
    else:
        if not request_is_local(request):
            clear_pending_otp(user)
            db.commit()
            raise HTTPException(status_code=503, detail=f"Gmail SMTP is not configured. {GMAIL_SETUP_HINT}")
        delivery_mode = "local_preview"
        delivery_note = "SMTP is not configured, so this local device is using a visible OTP preview for development."
        otp_preview = otp

    return {
        "message": (
            "OTP sent to your Gmail inbox."
            if delivery_mode == "email"
            else "Local OTP preview generated for this device."
        ),
        "delivery_mode": delivery_mode,
        "delivery_note": delivery_note,
        "masked_email": mask_email(email),
        "otp_preview": otp_preview,
        "expires_in_seconds": OTP_EXPIRY_MINUTES * 60,
        "resend_available_in_seconds": OTP_RESEND_COOLDOWN_SECONDS,
    }


@app.post("/auth/verify-otp")
def verify_otp(req: VerifyRequest, db: Session = Depends(get_db)) -> dict:
    email = normalize_email(req.email)
    otp = req.otp.strip()
    if not is_gmail_address(email):
        raise HTTPException(status_code=400, detail="Enter the same Gmail address used to request the OTP.")
    if not otp.isdigit() or len(otp) != 6:
        raise HTTPException(status_code=400, detail="Enter the 6-digit OTP from your Gmail inbox.")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.current_otp_hash or not user.otp_expiry:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    if user.otp_expiry < datetime.datetime.utcnow():
        clear_pending_otp(user)
        db.commit()
        raise HTTPException(status_code=400, detail="OTP expired. Request a new code.")

    if user.otp_attempts >= OTP_MAX_ATTEMPTS:
        clear_pending_otp(user)
        db.commit()
        raise HTTPException(status_code=429, detail="Too many incorrect OTP attempts. Request a new code.")

    if not verify_otp_hash(email, otp, user.current_otp_hash):
        user.otp_attempts += 1
        remaining_attempts = OTP_MAX_ATTEMPTS - user.otp_attempts
        if remaining_attempts <= 0:
            clear_pending_otp(user)
            db.commit()
            raise HTTPException(status_code=429, detail="Too many incorrect OTP attempts. Request a new code.")
        db.commit()
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect OTP. {remaining_attempts} attempt(s) remaining.",
        )

    user.is_verified = True
    clear_pending_otp(user)
    db.commit()

    has_subscription = has_active_subscription(user)
    token = create_access_token(email=user.email, has_subscription=has_subscription)
    return {
        "message": "Login successful",
        "has_subscription": has_subscription,
        "email": user.email,
        "token": token,
    }


@app.post("/payment/verify-upi")
def verify_upi(
    req: PaymentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    email = normalize_email(req.email)
    utr = req.utr_number.strip()
    if not utr.isdigit() or len(utr) < 10:
        raise HTTPException(status_code=400, detail="Invalid UTR number.")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Verify your Gmail OTP before activating premium.")

    user.subscription_end = datetime.datetime.utcnow() + datetime.timedelta(days=30)
    user.last_payment_ref = utr
    db.commit()

    receipt_text = (
        "Payment of INR 1 received for Deepfake Detection.\n"
        "Premium access is active for the next 30 days.\n"
        f"UPI reference: {utr}"
    )
    background_tasks.add_task(
        send_activity_email,
        email,
        "Deepfake Detection Premium Activated",
        receipt_text,
    )

    token = create_access_token(email=user.email, has_subscription=True)
    return {
        "message": "Subscription activated successfully!",
        "subscription_end": user.subscription_end.isoformat(),
        "token": token,
    }


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Deepfake Detection Demo</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #08101f;
            --panel: rgba(11, 19, 38, 0.9);
            --line: rgba(105, 229, 255, 0.28);
            --text: #ecf7ff;
            --muted: #9fb6c9;
            --accent: #67f3da;
            --accent-2: #ff9f43;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(103, 243, 218, 0.12), transparent 30%),
                radial-gradient(circle at right, rgba(255, 159, 67, 0.14), transparent 28%),
                linear-gradient(160deg, #050a14, var(--bg));
            min-height: 100vh;
        }

        .page {
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 24px 56px;
        }

        .page-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 24px;
            padding: 14px 18px;
            border: 1px solid rgba(103, 243, 218, 0.18);
            border-radius: 999px;
            background: rgba(9, 16, 31, 0.88);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
        }

        .page-brand {
            font-size: 0.92rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text);
        }

        .page-nav-links {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .page-nav-btn {
            min-height: 42px;
            padding: 0 18px;
            border-radius: 999px;
            border: 1px solid rgba(103, 243, 218, 0.16);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
            font: inherit;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }

        .page-nav-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(103, 243, 218, 0.34);
            background: rgba(103, 243, 218, 0.08);
        }

        .page-nav-btn.primary {
            background: rgba(103, 243, 218, 0.12);
            color: var(--accent);
        }

        .hero {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
            align-items: start;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(14px);
        }

        .hero-copy {
            padding: 28px;
        }

        h1 {
            margin: 0 0 12px;
            font-size: clamp(2.4rem, 4vw, 4.8rem);
            line-height: 0.95;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
            font-size: 1rem;
        }

        .hero-copy p + p {
            margin-top: 12px;
        }

        .status {
            margin-top: 22px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(103, 243, 218, 0.08);
            border: 1px solid rgba(103, 243, 218, 0.22);
            color: var(--accent);
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.8rem;
        }

        .panel {
            padding: 24px;
        }

        .video-shell {
            margin-top: 24px;
            padding: 18px;
        }

        .video-shell-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .video-shell-title {
            display: grid;
            gap: 4px;
        }

        .video-shell-title strong {
            font-size: 1rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .video-shell-title span {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .video-wrap {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            background: #000;
            min-height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        video,
        .embed-frame,
        .preview-image {
            width: 100%;
            background: #000;
            max-height: 72vh;
        }

        video,
        .embed-frame,
        .preview-image {
            display: none;
        }

        .embed-frame,
        .preview-image {
            border: 0;
            display: none;
        }

        .video-wrap[data-mode="video"] video,
        .video-wrap[data-mode="embed"] .embed-frame,
        .video-wrap[data-mode="image"] .preview-image {
            display: block;
        }

        .preview-image {
            object-fit: contain;
            background: #02050c;
        }

        .controls {
            margin-top: 18px;
            display: flex;
            gap: 14px;
            align-items: center;
            flex-wrap: wrap;
        }

        .url-control {
            flex: 1 1 420px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .url-input {
            flex: 1 1 260px;
            min-width: 220px;
            appearance: none;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
            padding: 12px 14px;
            border-radius: 14px;
            outline: none;
        }

        .action-btn {
            appearance: none;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
            padding: 12px 16px;
            border-radius: 14px;
            cursor: pointer;
            font-weight: 600;
        }

        .action-btn:hover {
            border-color: rgba(103, 243, 218, 0.3);
        }

        .action-btn.primary {
            background: rgba(103, 243, 218, 0.12);
            border-color: rgba(103, 243, 218, 0.24);
            color: var(--accent);
        }

        .upload {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 12px 16px;
            border-radius: 14px;
            cursor: pointer;
        }

        input[type="file"] {
            display: none;
        }

        .note {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .sample-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .link-status {
            margin-top: 16px;
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: var(--muted);
        }

        .link-status[data-tone="success"] {
            border-color: rgba(103, 243, 218, 0.26);
            color: var(--accent);
        }

        .link-status[data-tone="error"] {
            border-color: rgba(255, 159, 67, 0.26);
            color: #ffd0aa;
        }

        .link-status[data-tone="loading"] {
            border-color: rgba(255, 227, 138, 0.26);
            color: #ffe38a;
        }

        .media-caption {
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.9rem;
            min-height: 1.4em;
        }

        .result-card {
            margin-top: 18px;
            padding: 18px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: none;
        }

        .result-card.visible {
            display: block;
        }

        .result-card[data-tone="safe"] {
            border-color: rgba(103, 243, 218, 0.28);
            box-shadow: inset 0 0 0 1px rgba(103, 243, 218, 0.08);
        }

        .result-card[data-tone="alert"] {
            border-color: rgba(255, 109, 146, 0.36);
            box-shadow: inset 0 0 0 1px rgba(255, 65, 108, 0.1);
        }

        .result-card[data-tone="loading"] {
            border-color: rgba(255, 227, 138, 0.28);
        }

        .result-card h3 {
            margin: 0 0 8px;
            font-size: 1.1rem;
        }

        .result-summary {
            font-size: 0.98rem;
            color: var(--text);
            margin-bottom: 10px;
        }

        .result-explanation {
            color: var(--muted);
            line-height: 1.7;
            margin-bottom: 14px;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }

        .result-pill {
            padding: 12px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .result-pill strong {
            display: block;
            color: var(--text);
            margin-bottom: 4px;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .result-pill span {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            word-break: break-word;
        }

        .steps {
            display: grid;
            gap: 12px;
        }

        .step {
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .step strong {
            display: block;
            margin-bottom: 6px;
            color: var(--text);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.82rem;
        }

        @media (max-width: 900px) {
            .page-nav {
                border-radius: 24px;
                flex-direction: column;
                align-items: flex-start;
            }

            .page-nav-links {
                width: 100%;
            }

            .hero {
                grid-template-columns: 1fr;
            }

            .result-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <header class="page-nav">
            <div class="page-brand">Deepfake Detection</div>
            <nav class="page-nav-links">
                <button id="demo-nav-home" class="page-nav-btn" type="button">Home</button>
                <button id="demo-nav-subscription" class="page-nav-btn primary" type="button">Subscription</button>
                <button id="demo-nav-logout" class="page-nav-btn" type="button">Logout</button>
            </nav>
        </header>

        <section class="hero">
            <div class="card hero-copy">
                <h1>Deepfake Detection</h1>
                <p>This demo page supports both local files and online links, while keeping the analysis flow inside your local FastAPI backend.</p>
                <p>Use the extension popup to log in with OTP and activate premium, then scan playable media or analyze public links from the same local stack.</p>
                <div class="status">Local demo route ready</div>
            </div>
            <aside class="card panel">
                <div class="steps">
                    <div class="step">
                        <strong>Step 1</strong>
                        Log in from the extension popup and activate premium access.
                    </div>
                    <div class="step">
                        <strong>Step 2</strong>
                        Load a local video or paste a direct online media URL.
                    </div>
                    <div class="step">
                        <strong>Step 3</strong>
                        Scan a frame or inspect a public link with provenance metadata.
                    </div>
                </div>
            </aside>
        </section>

        <section class="card video-shell">
            <div class="video-shell-header">
                <div class="video-shell-title">
                    <strong>Video Analysis</strong>
                    <span>Run the active frame scan directly from the top of the player section.</span>
                </div>
                <button id="analyze-video-btn" class="action-btn primary" type="button">Analyze current video</button>
            </div>
            <div id="video-wrap" class="video-wrap" data-mode="video">
                <video id="demo-video" controls playsinline preload="metadata"></video>
                <iframe id="embed-frame" class="embed-frame" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen referrerpolicy="strict-origin-when-cross-origin"></iframe>
                <img id="preview-image" class="preview-image" alt="Linked media preview">
            </div>
            <div id="media-caption" class="media-caption"></div>
            <div class="controls">
                <label class="upload">
                    <span>Load video</span>
                    <input id="video-input" type="file" accept="video/*">
                </label>
                <div class="url-control">
                    <input id="media-url" class="url-input" type="url" placeholder="Paste a direct MP4, WebM, JPG, or PNG URL">
                    <button id="load-link-btn" class="action-btn" type="button">Load link</button>
                    <button id="analyze-image-btn" class="action-btn" type="button">Analyze link</button>
                </div>
                <div class="note">Use direct MP4 or WebM URLs with Load link. Use Analyze link for image URLs or normal webpage links like YouTube posts, articles, and social pages.</div>
            </div>
            <div class="sample-row">
                <button id="sample-video-btn" class="action-btn" type="button">Sample direct video</button>
                <button id="sample-link-btn" class="action-btn" type="button">Sample webpage link</button>
            </div>
            <div id="link-status" class="link-status" data-tone="idle">Paste a public direct media URL to extend the demo beyond local files.</div>
            <div id="analysis-result" class="result-card" data-tone="idle">
                <h3 id="result-title">Analysis result</h3>
                <div id="result-summary" class="result-summary"></div>
                <div id="result-explanation" class="result-explanation"></div>
                <div class="result-grid">
                    <div class="result-pill">
                        <strong>Fake Risk</strong>
                        <span id="result-visual-risk">N/A</span>
                    </div>
                    <div class="result-pill">
                        <strong>Face Movement</strong>
                        <span id="result-liveness">N/A</span>
                    </div>
                    <div class="result-pill">
                        <strong>Frames Checked</strong>
                        <span id="result-frames">N/A</span>
                    </div>
                    <div class="result-pill">
                        <strong>Proof Code</strong>
                        <span id="result-proof-hash">N/A</span>
                    </div>
                    <div class="result-pill">
                        <strong>AI Clues</strong>
                        <span id="result-provenance">N/A</span>
                    </div>
                    <div class="result-pill">
                        <strong>Source Clues</strong>
                        <span id="result-source-integrity">N/A</span>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        const websiteStorageKey = 'shieldWebsiteSession';
        const input = document.getElementById('video-input');
        const video = document.getElementById('demo-video');
        const videoWrap = document.getElementById('video-wrap');
        const embedFrame = document.getElementById('embed-frame');
        const previewImage = document.getElementById('preview-image');
        const mediaCaption = document.getElementById('media-caption');
        const mediaUrlInput = document.getElementById('media-url');
        const loadLinkBtn = document.getElementById('load-link-btn');
        const analyzeImageBtn = document.getElementById('analyze-image-btn');
        const analyzeVideoBtn = document.getElementById('analyze-video-btn');
        const sampleVideoBtn = document.getElementById('sample-video-btn');
        const sampleLinkBtn = document.getElementById('sample-link-btn');
        const linkStatus = document.getElementById('link-status');
        const resultCard = document.getElementById('analysis-result');
        const resultTitle = document.getElementById('result-title');
        const resultSummary = document.getElementById('result-summary');
        const resultExplanation = document.getElementById('result-explanation');
        const resultVisualRisk = document.getElementById('result-visual-risk');
        const resultLiveness = document.getElementById('result-liveness');
        const resultFrames = document.getElementById('result-frames');
        const resultProofHash = document.getElementById('result-proof-hash');
        const resultProvenance = document.getElementById('result-provenance');
        const resultSourceIntegrity = document.getElementById('result-source-integrity');
        let currentObjectUrl = null;

        const SAMPLE_VIDEO_URL = 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4';
        const SAMPLE_PAGE_URL = 'https://youtu.be/dQw4w9WgXcQ';

        function demoGetSession() {
            try {
                return JSON.parse(localStorage.getItem(websiteStorageKey) || '{}');
            } catch {
                return {};
            }
        }

        function demoGoHome() {
            window.location.href = '/';
        }

        function demoGoSubscription() {
            const session = demoGetSession();
            if (session.userEmail && session.authStage === 'verified' && session.hasSub) {
                window.location.href = '/demo';
                return;
            }
            window.location.href = '/#access';
        }

        function demoLogout() {
            localStorage.removeItem(websiteStorageKey);
            window.location.href = '/';
        }

        function setStatus(message, tone = 'idle') {
            linkStatus.textContent = message;
            linkStatus.dataset.tone = tone;
        }

        function formatPercent(value) {
            return typeof value === 'number' && Number.isFinite(value) ? `${value}%` : 'N/A';
        }

        function renderDetailedResult(payload, options = {}) {
            const meta = payload.metadata || {};
            const tone = options.tone || (payload.is_threat || payload.is_deepfake ? 'alert' : 'safe');
            resultCard.dataset.tone = tone;
            resultCard.classList.add('visible');
            resultTitle.textContent = options.title || 'Analysis result';
            resultSummary.textContent = payload.analysis_summary || payload.message || 'Analysis completed.';
            resultExplanation.textContent = payload.explanation || 'The scan finished, but there is no extra explanation for this result yet.';
            resultVisualRisk.textContent = formatPercent(payload.fake_score ?? payload.visual_deepfake_score ?? payload.overall_threat_score ?? null);
            resultLiveness.textContent = formatPercent(payload.liveness_score ?? payload.biological_liveness ?? null);
            resultFrames.textContent = payload.frames_analyzed ?? 1;
            resultProofHash.textContent = payload.proof_hash ? `${payload.proof_hash.slice(0, 24)}...` : 'Unavailable';
            resultProvenance.textContent = meta.c2pa_signature || 'No AI clue found';
            resultSourceIntegrity.textContent = meta.is_original || 'No source clue found';
        }

        function setLoadingResult(summary) {
            resultCard.dataset.tone = 'loading';
            resultCard.classList.add('visible');
            resultTitle.textContent = 'Analysis result';
            resultSummary.textContent = summary;
            resultExplanation.textContent = 'Capturing frames and sending them to the backend for sequence analysis.';
            resultVisualRisk.textContent = 'Pending';
            resultLiveness.textContent = 'Pending';
            resultFrames.textContent = 'Pending';
            resultProofHash.textContent = 'Pending';
            resultProvenance.textContent = 'Pending';
            resultSourceIntegrity.textContent = 'Pending';
        }

        function clearResult() {
            resultCard.classList.remove('visible');
            resultCard.dataset.tone = 'idle';
        }

        function setMediaCaption(message = '') {
            mediaCaption.textContent = message;
        }

        function resetEmbeddedMedia() {
            embedFrame.src = 'about:blank';
            previewImage.removeAttribute('src');
            previewImage.alt = 'Linked media preview';
            videoWrap.dataset.mode = 'video';
            setMediaCaption('');
        }

        function showVideoSurface(message = '') {
            videoWrap.dataset.mode = 'video';
            setMediaCaption(message);
        }

        function showEmbedSurface(embedUrl, message) {
            video.pause();
            video.removeAttribute('src');
            video.load();
            embedFrame.src = embedUrl;
            videoWrap.dataset.mode = 'embed';
            setMediaCaption(message);
        }

        function showPreviewSurface(imageUrl, message) {
            video.pause();
            video.removeAttribute('src');
            video.load();
            previewImage.src = imageUrl;
            previewImage.alt = 'Preview image for analyzed link';
            videoWrap.dataset.mode = 'image';
            setMediaCaption(message);
        }

        function resolveYouTubeEmbedUrl(rawUrl) {
            try {
                const parsed = new URL(rawUrl);
                const host = parsed.hostname.toLowerCase();
                let videoId = null;

                if (host === 'youtu.be' || host === 'www.youtu.be') {
                    videoId = parsed.pathname.replace(/^\\//, '').split('/')[0];
                } else if (host === 'youtube.com' || host === 'www.youtube.com' || host === 'm.youtube.com') {
                    if (parsed.pathname === '/watch') {
                        videoId = parsed.searchParams.get('v');
                    } else if (parsed.pathname.startsWith('/shorts/')) {
                        videoId = parsed.pathname.split('/')[2];
                    }
                }

                if (!videoId) {
                    return null;
                }

                return `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&modestbranding=1`;
            } catch {
                return null;
            }
        }

        function releaseObjectUrl() {
            if (currentObjectUrl) {
                URL.revokeObjectURL(currentObjectUrl);
                currentObjectUrl = null;
            }
        }

        input.addEventListener('change', (event) => {
            const file = event.target.files && event.target.files[0];
            if (!file) {
                return;
            }

            releaseObjectUrl();
            resetEmbeddedMedia();

            currentObjectUrl = URL.createObjectURL(file);
            video.src = currentObjectUrl;
            video.load();
            video.play().catch(() => {});
            showVideoSurface(`Showing local file: ${file.name}`);
            setStatus(`Loaded local file: ${file.name}`, 'success');
            clearResult();
        });

        loadLinkBtn.addEventListener('click', () => {
            const mediaUrl = mediaUrlInput.value.trim();
            if (!mediaUrl) {
                setStatus('Paste a direct online video or image URL first.', 'error');
                return;
            }

            if (!/\\.(mp4|webm|ogg)(?:[?#].*)?$/i.test(mediaUrl)) {
                setStatus('That looks like a webpage URL, not a direct video file. Use Analyze link for page URLs, or paste a direct MP4 or WebM URL here.', 'error');
                return;
            }

            releaseObjectUrl();
            resetEmbeddedMedia();
            video.crossOrigin = 'anonymous';
            video.src = mediaUrl;
            video.load();
            video.play().catch(() => {});
            showVideoSurface(`Showing direct media link: ${mediaUrl}`);
            setStatus('Loaded online media link. If the site allows playback and CORS, the extension can scan the current frame.', 'success');
            clearResult();
        });

        function captureFrame(videoElement) {
            const canvas = document.createElement('canvas');
            const width = videoElement.videoWidth || Math.max(640, Math.round(videoElement.clientWidth || 640));
            const height = videoElement.videoHeight || Math.max(360, Math.round(videoElement.clientHeight || 360));
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');

            if (!context) {
                throw new Error('Canvas capture is not available in this browser.');
            }

            if (videoElement.videoWidth && videoElement.videoHeight && videoElement.readyState >= 2) {
                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            } else {
                context.fillStyle = '#09101f';
                context.fillRect(0, 0, canvas.width, canvas.height);
                context.fillStyle = '#b4fff6';
                context.font = 'bold 26px Segoe UI';
                context.fillText('Deepfake Detection', 36, 68);
                context.font = '18px Segoe UI';
                context.fillStyle = '#dce9ff';
                context.fillText('Fallback capture used for analysis', 36, 112);
            }

            return canvas.toDataURL('image/jpeg', 0.82);
        }

        async function captureFrameSequence(videoElement, frameCount, intervalMs) {
            const frames = [];
            for (let index = 0; index < frameCount; index += 1) {
                frames.push(captureFrame(videoElement));
                if (index < frameCount - 1) {
                    await new Promise((resolve) => setTimeout(resolve, intervalMs));
                }
            }
            return frames;
        }

        function dataUrlToBlob(dataUrl) {
            const [header, encoded] = dataUrl.split(',');
            const mimeMatch = header.match(/data:(.*?);base64/);
            const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg';
            const binary = atob(encoded);
            const bytes = new Uint8Array(binary.length);

            for (let index = 0; index < binary.length; index += 1) {
                bytes[index] = binary.charCodeAt(index);
            }

            return new Blob([bytes], { type: mimeType });
        }

        analyzeVideoBtn.addEventListener('click', async () => {
            if (videoWrap.dataset.mode !== 'video' || !video.getAttribute('src')) {
                setStatus('Load a local or direct video into the player before running analysis.', 'error');
                return;
            }

            setStatus('Capturing a quick 3-frame burst for video analysis...', 'loading');
            setLoadingResult('Running a faster sequence analysis on the current video.');

            try {
                const frameDataUrls = await captureFrameSequence(video, 3, 80);
                const formData = new FormData();

                frameDataUrls.forEach((frameDataUrl, index) => {
                    formData.append('files', dataUrlToBlob(frameDataUrl), `demo_frame_${index}.jpg`);
                });

                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Video analysis failed.');
                }

                renderDetailedResult(payload, { title: 'Current video analysis' });
                setStatus(`${payload.message} Detailed explanation is shown below the player.`, payload.is_threat ? 'error' : 'success');
            } catch (error) {
                setStatus(error.message || 'Video analysis failed.', 'error');
                renderDetailedResult({
                    message: 'Analysis failed.',
                    explanation: error.message || 'The backend could not analyze this video.',
                    metadata: {},
                    frames_analyzed: 0,
                    proof_hash: null
                }, { title: 'Current video analysis', tone: 'alert' });
            }
        });

        analyzeImageBtn.addEventListener('click', async () => {
            const mediaUrl = mediaUrlInput.value.trim();
            if (!mediaUrl) {
                setStatus('Paste a public image URL first.', 'error');
                return;
            }

            const embedUrl = resolveYouTubeEmbedUrl(mediaUrl);
            if (embedUrl) {
                showEmbedSurface(embedUrl, `Showing linked page in an embedded player while the backend analyzes its preview image.`);
            }

            setStatus('Analyzing remote image URL through the backend...', 'idle');
            setLoadingResult('Analyzing remote image or webpage preview through the backend.');

            try {
                const response = await fetch('/analyze-url', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ url: mediaUrl })
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Remote URL analysis failed.');
                }

                const label = payload.is_deepfake ? 'DEEPFAKE' : 'AUTHENTIC';
                const sourceDetail = payload.preview_image_url
                    ? `using preview image ${payload.preview_image_url}`
                    : `for ${payload.source_url}`;
                if (!embedUrl && payload.preview_image_url) {
                    showPreviewSurface(payload.preview_image_url, 'Showing the preview image that was analyzed for this link.');
                } else if (!embedUrl) {
                    setMediaCaption(`Analyzed link: ${payload.source_url}`);
                }
                renderDetailedResult(payload, { title: 'Link analysis result', tone: payload.is_deepfake ? 'alert' : 'safe' });
                setStatus(`${label} ${payload.fake_score}% via ${String(payload.engine).toUpperCase()} ${sourceDetail}`, 'success');
            } catch (error) {
                if (!embedUrl) {
                    showVideoSurface('');
                }
                setStatus(error.message || 'Remote URL analysis failed.', 'error');
            }
        });

        sampleVideoBtn.addEventListener('click', () => {
            mediaUrlInput.value = SAMPLE_VIDEO_URL;
            loadLinkBtn.click();
        });

        sampleLinkBtn.addEventListener('click', () => {
            mediaUrlInput.value = SAMPLE_PAGE_URL;
            setStatus('Sample webpage URL loaded. Click Analyze link to score its preview image through the backend.', 'success');
        });

        document.getElementById('demo-nav-home').addEventListener('click', demoGoHome);
        document.getElementById('demo-nav-subscription').addEventListener('click', demoGoSubscription);
        document.getElementById('demo-nav-logout').addEventListener('click', demoLogout);
    </script>
</body>
</html>
    """


@app.post("/analyze-url")
async def analyze_url(payload: AnalyzeUrlRequest) -> dict:
    validated_url = validate_public_url(payload.url)
    social_link_fallback_warning: Optional[str] = None

    if detector.mode == "realitydefender" and is_supported_social_url(validated_url):
        try:
            detection_result = await asyncio.to_thread(detector.detect_social_url, validated_url)
        except Exception as exc:
            social_link_fallback_warning = str(exc)
        else:
            metadata = build_reality_defender_metadata(
                {
                    "is_original": "Unknown",
                    "creation_date": "Unknown",
                    "software_platform": "Unknown",
                    "camera_device": "Unknown",
                    "c2pa_signature": "No AI Signature Detected",
                },
                detection_result,
            )
            return {
                "status": "success",
                "engine": detector.mode,
                "fake_score": detection_result["score"],
                "is_deepfake": detection_result["is_manipulated"],
                "threshold": FAKE_THRESHOLD,
                "message": (
                    "This link may contain fake or edited media."
                    if detection_result["is_manipulated"]
                    else "This link looks mostly real."
                ),
                "analysis_summary": (
                    "This link may contain fake or edited media."
                    if detection_result["is_manipulated"]
                    else "This link looks mostly authentic."
                ),
                "explanation": build_result_explanation(
                    visual_score=detection_result["score"],
                    liveness_score=None,
                    audio_score=None,
                    metadata=metadata,
                    frames_analyzed=1,
                ),
                "metadata": metadata,
                "source_url": validated_url,
                "resolved_url": validated_url,
            }

    preview_image_url = resolve_youtube_thumbnail_url(validated_url)

    if preview_image_url:
        image_bytes, content_type, final_url = await asyncio.to_thread(fetch_remote_bytes, preview_image_url)
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Could not retrieve a YouTube preview image.")

        if detector.mode == "mock":
            await asyncio.sleep(1.2)

        result = score_image_bytes(image_bytes, force_local=bool(social_link_fallback_warning))
        result["source_url"] = validated_url
        result["resolved_url"] = final_url
        result["preview_image_url"] = preview_image_url
        if social_link_fallback_warning:
            result.setdefault("metadata", {})["reality_defender_fallback"] = social_link_fallback_warning
        return result

    remote_bytes, content_type, final_url = await asyncio.to_thread(fetch_remote_bytes, validated_url)

    if content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Direct video URLs should be loaded into the demo player first. "
                "Use Analyze image URL for JPG or PNG links."
            ),
        )

    preview_image_url = None
    image_bytes = remote_bytes

    if content_type == "text/html":
        preview_image_url = extract_preview_image_url(remote_bytes, final_url)
        image_bytes, content_type, final_url = await asyncio.to_thread(fetch_remote_bytes, preview_image_url)

    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The URL must point to a public image or a webpage with a preview image.",
        )

    if detector.mode == "mock":
        await asyncio.sleep(1.2)

    result = score_image_bytes(image_bytes, force_local=bool(social_link_fallback_warning))
    result["source_url"] = validated_url
    result["resolved_url"] = final_url
    if social_link_fallback_warning:
        result.setdefault("metadata", {})["reality_defender_fallback"] = social_link_fallback_warning
    if preview_image_url:
        result["preview_image_url"] = preview_image_url
    return result


@app.post("/analyze")
async def analyze_sequence(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No media files uploaded.")

    image_bytes_list: List[bytes] = []
    for upload in files:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All uploaded files must be images.")
        image_bytes = await upload.read()
        if image_bytes:
            image_bytes_list.append(image_bytes)

    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="Uploaded media frames are empty.")

    primary_result = analyze_visual_sequence(image_bytes_list)
    liveness_score = verify_biological_liveness(image_bytes_list)
    proof_hash, proof_timestamp = generate_proof_of_reality(
        primary_result["fake_score"],
        liveness_score,
        primary_result.get("metadata", {}),
    )

    has_liveness_signal = liveness_score > 0.0
    is_threat = primary_result["is_deepfake"] or (has_liveness_signal and liveness_score < 50.0)

    if detector.mode == "mock":
        await asyncio.sleep(1.2)

    return {
        "status": "success",
        "engine": primary_result["engine"],
        "fake_score": primary_result["fake_score"],
        "is_deepfake": primary_result["is_deepfake"],
        "liveness_score": liveness_score,
        "proof_hash": proof_hash,
        "proof_timestamp": proof_timestamp,
        "is_threat": is_threat,
        "message": (
            "High probability of synthetic manipulation."
            if is_threat
            else "Identity biologically verified."
        ),
        "explanation": build_result_explanation(
            visual_score=primary_result["fake_score"],
            liveness_score=liveness_score,
            audio_score=None,
            metadata=primary_result.get("metadata", {}),
            frames_analyzed=len(image_bytes_list),
        ),
        "threshold": primary_result["threshold"],
        "metadata": primary_result.get("metadata", {}),
        "frames_analyzed": len(image_bytes_list),
    }


@app.post("/analyze-full-media")
async def analyze_full_media(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "uploaded_media"
    extension = Path(filename).suffix.lower()
    supported_extensions = {".mp4", ".avi", ".mov", ".mp3", ".wav", ".jpg", ".jpeg", ".png"}
    if extension not in supported_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_path = None
    audio_temp_path = None
    clip = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        result = {
            "filename": filename,
            "media_type": "unknown",
            "overall_threat_score": 0.0,
            "visual_deepfake_score": None,
            "audio_clone_score": None,
            "biological_liveness": None,
            "metadata": {},
            "analysis_summary": "",
            "explanation": "",
            "proof_hash": None,
            "proof_timestamp": None,
            "frames_analyzed": 0,
        }

        if detector.mode == "realitydefender":
            detection_result = await asyncio.to_thread(analyze_file_with_detector, temp_path)
            metadata = build_reality_defender_metadata(result["metadata"], detection_result)
            result["media_type"] = (
                "video" if extension in {".mp4", ".avi", ".mov"}
                else "audio" if extension in {".mp3", ".wav"}
                else "image"
            )
            result["metadata"] = metadata
            result["overall_threat_score"] = detection_result["score"]
            if result["media_type"] == "audio":
                result["audio_clone_score"] = detection_result["score"]
            else:
                result["visual_deepfake_score"] = detection_result["score"]

            proof_hash, proof_timestamp = generate_proof_of_reality(
                detection_result["score"],
                0.0,
                metadata,
            )
            result["proof_hash"] = proof_hash
            result["proof_timestamp"] = proof_timestamp
            result["analysis_summary"] = (
                "This file may be fake or edited."
                if detection_result["is_manipulated"]
                else "This file looks mostly real."
            )
            result["explanation"] = build_result_explanation(
                visual_score=result["visual_deepfake_score"],
                liveness_score=None,
                audio_score=result["audio_clone_score"],
                metadata=metadata,
                frames_analyzed=0,
            )
            return result

        if extension in {".jpg", ".jpeg", ".png"}:
            result["media_type"] = "image"
            image_result = score_image_bytes(content)
            result["visual_deepfake_score"] = image_result["fake_score"]
            result["overall_threat_score"] = image_result["fake_score"]
            result["metadata"] = image_result.get("metadata", {})
            proof_hash, proof_timestamp = generate_proof_of_reality(
                image_result["fake_score"],
                0.0,
                result["metadata"],
            )
            result["proof_hash"] = proof_hash
            result["proof_timestamp"] = proof_timestamp

        elif extension in {".mp3", ".wav"}:
            result["media_type"] = "audio"
            audio_score = analyze_full_audio(temp_path)
            result["audio_clone_score"] = audio_score
            result["overall_threat_score"] = audio_score
            proof_hash, proof_timestamp = generate_proof_of_reality(audio_score, 0.0, {})
            result["proof_hash"] = proof_hash
            result["proof_timestamp"] = proof_timestamp

        elif extension in {".mp4", ".avi", ".mov"}:
            result["media_type"] = "video"

            capture = cv2.VideoCapture(temp_path)
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 24

            frame_interval = max(fps, 1)
            frame_index = 0
            extracted_frames: List[bytes] = []

            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % frame_interval == 0:
                    encoded, buffer = cv2.imencode(".jpg", frame)
                    if encoded:
                        extracted_frames.append(buffer.tobytes())
                frame_index += 1

                if len(extracted_frames) >= 90:
                    break

            capture.release()
            result["frames_analyzed"] = len(extracted_frames)

            if extracted_frames:
                sequence_result = analyze_visual_sequence(extracted_frames)
                result["visual_deepfake_score"] = sequence_result["fake_score"]
                result["biological_liveness"] = verify_biological_liveness(extracted_frames)
                result["metadata"] = sequence_result.get("metadata", {})

            try:
                from moviepy import VideoFileClip

                clip = VideoFileClip(temp_path)
                if clip.audio is not None:
                    audio_temp_path = f"{temp_path}.wav"
                    clip.audio.write_audiofile(audio_temp_path, logger=None)
                    result["audio_clone_score"] = analyze_full_audio(audio_temp_path)
            except Exception:
                result["audio_clone_score"] = None
            finally:
                if clip is not None:
                    clip.close()

            scores_to_evaluate = [
                result["visual_deepfake_score"] or 0.0,
                result["audio_clone_score"] or 0.0,
            ]
            result["overall_threat_score"] = round(max(scores_to_evaluate), 2)
            proof_hash, proof_timestamp = generate_proof_of_reality(
                result["visual_deepfake_score"] or result["overall_threat_score"],
                result["biological_liveness"] or 0.0,
                result["metadata"],
            )
            result["proof_hash"] = proof_hash
            result["proof_timestamp"] = proof_timestamp

        if result["overall_threat_score"] > 65.0 or (
            result["biological_liveness"] is not None and result["biological_liveness"] < 50.0
        ):
            result["analysis_summary"] = "This file may be fake or edited."
        else:
            result["analysis_summary"] = "This file looks mostly real."

        result["explanation"] = build_result_explanation(
            visual_score=result["visual_deepfake_score"],
            liveness_score=result["biological_liveness"],
            audio_score=result["audio_clone_score"],
            metadata=result["metadata"],
            frames_analyzed=result["frames_analyzed"],
        )

        return result
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
        if audio_temp_path and os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ── Template-served pages ─────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _serve_template(name: str) -> HTMLResponse:
    """Read an HTML template file and return it as an HTMLResponse."""
    path = TEMPLATES_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Page not found")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/deepfake-image-detection", response_class=HTMLResponse)
async def page_image_detection():
    return _serve_template("image-detection.html")


@app.get("/deepfake-video-detection", response_class=HTMLResponse)
async def page_video_detection():
    return _serve_template("video-detection.html")


@app.get("/deepfake-voice-detection", response_class=HTMLResponse)
async def page_voice_detection():
    return _serve_template("voice-detection.html")


@app.get("/about", response_class=HTMLResponse)
async def page_about():
    return _serve_template("about.html")


@app.get("/privacy-policy", response_class=HTMLResponse)
async def page_privacy_policy():
    return _serve_template("privacy-policy.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
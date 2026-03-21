import asyncio
import html
import hashlib
import ipaddress
import io
import os
import re
import socket
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


app = FastAPI(
    title="Synthetic Media Shield API",
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


def build_detector() -> tuple[object, Optional[str]]:
    if not MODEL_PATH.exists():
        return MockDetector(), f"Model file not found at {MODEL_PATH}. Using mock detector."

    try:
        return TorchDetector(MODEL_PATH), None
    except Exception as exc:  # pragma: no cover - fallback path for demo resilience
        return MockDetector(), f"Failed to load model: {exc}. Using mock detector."


detector, startup_warning = build_detector()


class AnalyzeUrlRequest(BaseModel):
    url: str


def score_image_bytes(image_bytes: bytes) -> dict:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc

    fake_score = detector.predict(image, image_bytes)
    is_deepfake = fake_score > FAKE_THRESHOLD

    return {
        "status": "success",
        "engine": detector.mode,
        "fake_score": fake_score,
        "is_deepfake": is_deepfake,
        "threshold": FAKE_THRESHOLD,
        "message": (
            "High probability of synthetic manipulation detected."
            if is_deepfake
            else "Media appears authentic."
        ),
    }


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
    request = Request(url, headers={"User-Agent": "SyntheticMediaShield/1.0"})

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


@app.get("/")
def root() -> dict:
    return {
        "service": "Synthetic Media Shield API",
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
    }


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Synthetic Media Shield Demo</title>
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

        .media-caption {
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.9rem;
            min-height: 1.4em;
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
            .hero {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <div class="card hero-copy">
                <h1>Synthetic Media Shield</h1>
                <p>This demo page supports both local files and online links, while keeping the analysis flow inside your local FastAPI backend.</p>
                <p>Use a local video for the most reliable live demo. For public internet media, paste a direct MP4, WebM, or image URL into the link box below.</p>
                <div class="status">Local demo route ready</div>
            </div>
            <aside class="card panel">
                <div class="steps">
                    <div class="step">
                        <strong>Step 1</strong>
                        Load a local video or paste a direct online media URL.
                    </div>
                    <div class="step">
                        <strong>Step 2</strong>
                        For videos, play or scrub to the frame you want to inspect.
                    </div>
                    <div class="step">
                        <strong>Step 3</strong>
                        Click the extension's SCAN MEDIA button or analyze a remote image URL directly.
                    </div>
                </div>
            </aside>
        </section>

        <section class="card video-shell">
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
        </section>
    </main>

    <script>
        const input = document.getElementById('video-input');
        const video = document.getElementById('demo-video');
        const videoWrap = document.getElementById('video-wrap');
        const embedFrame = document.getElementById('embed-frame');
        const previewImage = document.getElementById('preview-image');
        const mediaCaption = document.getElementById('media-caption');
        const mediaUrlInput = document.getElementById('media-url');
        const loadLinkBtn = document.getElementById('load-link-btn');
        const analyzeImageBtn = document.getElementById('analyze-image-btn');
        const sampleVideoBtn = document.getElementById('sample-video-btn');
        const sampleLinkBtn = document.getElementById('sample-link-btn');
        const linkStatus = document.getElementById('link-status');
        let currentObjectUrl = null;

        const SAMPLE_VIDEO_URL = 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4';
        const SAMPLE_PAGE_URL = 'https://youtu.be/dQw4w9WgXcQ';

        function setStatus(message, tone = 'idle') {
            linkStatus.textContent = message;
            linkStatus.dataset.tone = tone;
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

            try {
                const response = await fetch('/analyze-url', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ url: mediaUrl })
                });

                const payload = await response.json();
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
    </script>
</body>
</html>
    """


@app.post("/analyze-url")
async def analyze_url(payload: AnalyzeUrlRequest) -> dict:
    validated_url = validate_public_url(payload.url)
    preview_image_url = resolve_youtube_thumbnail_url(validated_url)

    if preview_image_url:
        image_bytes, content_type, final_url = await asyncio.to_thread(fetch_remote_bytes, preview_image_url)
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Could not retrieve a YouTube preview image.")

        if detector.mode == "mock":
            await asyncio.sleep(1.2)

        result = score_image_bytes(image_bytes)
        result["source_url"] = validated_url
        result["resolved_url"] = final_url
        result["preview_image_url"] = preview_image_url
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

    result = score_image_bytes(image_bytes)
    result["source_url"] = validated_url
    result["resolved_url"] = final_url
    if preview_image_url:
        result["preview_image_url"] = preview_image_url
    return result


@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    if detector.mode == "mock":
        await asyncio.sleep(1.2)

    return score_image_bytes(image_bytes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
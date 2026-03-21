# Synthetic Media Shield

Synthetic Media Shield is a local demo stack for scanning visible video frames with a Chrome extension and a FastAPI backend.

## Project layout

```text
deepfake-shield/
├── backend/
│   ├── main.py
│   ├── model.pt
│   └── requirements.txt
├── extension/
│   ├── background.js
│   ├── content.js
│   ├── manifest.json
│   └── styles.css
└── README.md
```

## Backend behavior

- If `backend/model.pt` exists and loads correctly, the API uses the PyTorch model.
- If the model file is missing or fails to load, the API falls back to a deterministic mock engine so the UI still works.
- `GET /health` returns engine status for a quick smoke test.
- SQLite-backed OTP login and premium subscription state are stored in `backend/shield_saas.db`.

## SaaS flow

- Open the extension popup from the browser toolbar.
- Request an OTP using your email.
- If SMTP is configured through environment variables, the OTP is emailed. If not, the popup shows a local demo OTP preview so the hackathon flow still works.
- After OTP verification, activate premium with a UTR/reference number to unlock scans.
- Scan results now include forensic metadata such as EXIF presence, software tags, device hints, and AI provenance markers.

## SMTP configuration

- Set `SMS_SHIELD_SMTP_EMAIL` and `SMS_SHIELD_SMTP_PASSWORD` before starting the backend if you want real Gmail OTP delivery.
- Optional variables: `SMS_SHIELD_SMTP_SERVER`, `SMS_SHIELD_SMTP_PORT`, and `SMS_SHIELD_JWT_SECRET`.
- Without SMTP configuration, the app remains fully demoable through a local OTP preview in the popup.

## Local startup

1. Open a terminal in `deepfake-shield/backend`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the API with `uvicorn main:app --reload`.
4. Open `chrome://extensions`, enable Developer mode, and load the `deepfake-shield/extension` folder.

## Quick launch scripts

- Run `deepfake-shield/start-backend.ps1` to start the FastAPI service from the project virtual environment.
- Run `deepfake-shield/launch-extension.ps1` to open Chrome, Edge, or Brave with the unpacked extension preloaded into a dedicated browser profile and land on the local demo page.

## Reliable demo flow

- Open `http://127.0.0.1:8000/demo` in the browser that has the extension loaded.
- Load a local video file into the page.
- Click the extension's `SCAN MEDIA` button over the video.
- This demo path avoids the browser security restrictions that often block canvas capture on sites like YouTube, TikTok, and X.

## Online link flow

- Paste a direct public media URL into the demo page.
- For direct MP4 or WebM links, click `Load link`, then use the extension overlay to scan the current frame.
- For direct JPG or PNG links, click `Analyze image URL` and the backend will fetch and score the image itself.
- Remote URL analysis is restricted to public `http(s)` hosts and capped to small files for safety.

## Optional model dependencies

- The demo runs immediately with the mock engine and does not require PyTorch.
- If you want to load a real model later, install `backend/requirements-model.txt` in a Python version supported by PyTorch, then place `model.pt` in `backend/`.

## Real model swap

Drop your trained file at `backend/model.pt` or set the `DEEPFAKE_MODEL_PATH` environment variable before starting the API.
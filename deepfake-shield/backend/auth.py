import datetime
import hashlib
import hmac
import os
import re
import secrets
import smtplib
import ssl
from pathlib import Path
from email.message import EmailMessage
from email.utils import parseaddr

import jwt


def load_env_file() -> None:
    candidate_paths = [
        Path(__file__).with_name(".env"),
        Path(__file__).resolve().parent.parent / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()


def get_env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


SMTP_SERVER = get_env_value("SMS_SHIELD_SMTP_SERVER", "SMTP_SERVER", default="smtp.gmail.com")
SMTP_PORT = int(get_env_value("SMS_SHIELD_SMTP_PORT", "SMTP_PORT", default="587"))
SENDER_EMAIL = get_env_value("SMS_SHIELD_SMTP_EMAIL", "EMAIL_USER")
SENDER_PASSWORD = get_env_value("SMS_SHIELD_SMTP_PASSWORD", "EMAIL_PASS")
JWT_SECRET = os.getenv("SMS_SHIELD_JWT_SECRET", "hackcraft-local-dev-secret")
JWT_ALGORITHM = "HS256"
OTP_HASH_SECRET = os.getenv("SMS_SHIELD_OTP_SECRET", JWT_SECRET)
OTP_EXPIRY_MINUTES = int(os.getenv("SMS_SHIELD_OTP_EXPIRY_MINUTES", "10"))
OTP_RESEND_COOLDOWN_SECONDS = int(os.getenv("SMS_SHIELD_OTP_RESEND_COOLDOWN_SECONDS", "60"))
OTP_MAX_ATTEMPTS = int(os.getenv("SMS_SHIELD_OTP_MAX_ATTEMPTS", "5"))
SMTP_TIMEOUT_SECONDS = int(os.getenv("SMS_SHIELD_SMTP_TIMEOUT_SECONDS", "20"))
GMAIL_DOMAIN = "gmail.com"
GMAIL_SETUP_HINT = (
    "Set SMS_SHIELD_SMTP_EMAIL to your Gmail address and "
    "SMS_SHIELD_SMTP_PASSWORD to a Gmail App Password."
)
GMAIL_LOCAL_PART_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9._%+\-]{0,62}[a-z0-9])?$",
    re.IGNORECASE,
)


def email_delivery_configured() -> bool:
    return bool(SENDER_EMAIL and SENDER_PASSWORD)


def normalize_email(email: str) -> str:
    return parseaddr(email or "")[1].strip().lower()


def is_gmail_address(email: str) -> bool:
    local_part, separator, domain = normalize_email(email).partition("@")
    return bool(
        separator
        and domain == GMAIL_DOMAIN
        and local_part
        and len(local_part) <= 64
        and GMAIL_LOCAL_PART_PATTERN.fullmatch(local_part)
    )


def mask_email(email: str) -> str:
    normalized = normalize_email(email)
    local_part, _, domain = normalized.partition("@")
    if len(local_part) <= 2:
        masked_local = f"{local_part[:1]}*"
    else:
        masked_local = f"{local_part[:2]}{'*' * max(len(local_part) - 3, 1)}{local_part[-1]}"
    return f"{masked_local}@{domain}"


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(email: str, otp: str) -> str:
    payload = f"{normalize_email(email)}:{otp}:{OTP_HASH_SECRET}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_otp_hash(email: str, otp: str, expected_hash: str) -> bool:
    if not expected_hash:
        return False
    candidate_hash = hash_otp(email, otp)
    return hmac.compare_digest(candidate_hash, expected_hash)


def send_otp_email(receiver_email: str, otp: str, activity: str = "Login Attempt") -> tuple[bool, str]:
    message_body = (
        f"Activity: {activity}\n\n"
        f"Your Synthetic Media Shield OTP is: {otp}\n"
        f"This expires in {OTP_EXPIRY_MINUTES} minutes.\n"
        "If you did not request this code, you can ignore this email."
    )
    return send_activity_email(
        receiver_email=receiver_email,
        subject=f"Synthetic Media Shield OTP: {activity}",
        message_body=message_body,
    )


def send_activity_email(receiver_email: str, subject: str, message_body: str) -> tuple[bool, str]:
    if not email_delivery_configured():
        return False, f"Gmail SMTP is not configured. {GMAIL_SETUP_HINT}"

    message = EmailMessage()
    message.set_content(message_body)
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(
                SMTP_SERVER,
                SMTP_PORT,
                timeout=SMTP_TIMEOUT_SECONDS,
                context=ssl.create_default_context(),
            ) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(message)
        return True, "Email sent successfully."
    except Exception as exc:  # pragma: no cover - depends on external SMTP config
        return False, str(exc)


def create_access_token(email: str, has_subscription: bool) -> str:
    payload = {
        "sub": email,
        "has_subscription": has_subscription,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
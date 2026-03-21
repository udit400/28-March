import datetime
import os
import random
import smtplib
from email.message import EmailMessage

import jwt


SMTP_SERVER = os.getenv("SMS_SHIELD_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMS_SHIELD_SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SMS_SHIELD_SMTP_EMAIL", "")
SENDER_PASSWORD = os.getenv("SMS_SHIELD_SMTP_PASSWORD", "")
JWT_SECRET = os.getenv("SMS_SHIELD_JWT_SECRET", "hackcraft-local-dev-secret")
JWT_ALGORITHM = "HS256"


def email_delivery_configured() -> bool:
    return bool(SENDER_EMAIL and SENDER_PASSWORD)


def generate_otp() -> str:
    return f"{random.randint(100000, 999999):06d}"


def send_otp_email(receiver_email: str, otp: str, activity: str = "Login Attempt") -> tuple[bool, str]:
    message_body = (
        f"Activity: {activity}\n\n"
        f"Your Synthetic Media Shield OTP is: {otp}\n"
        "This expires in 10 minutes."
    )
    return send_activity_email(
        receiver_email=receiver_email,
        subject=f"Synthetic Media Shield: {activity}",
        message_body=message_body,
    )


def send_activity_email(receiver_email: str, subject: str, message_body: str) -> tuple[bool, str]:
    if not email_delivery_configured():
        return False, "SMTP credentials are not configured."

    message = EmailMessage()
    message.set_content(message_body)
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
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
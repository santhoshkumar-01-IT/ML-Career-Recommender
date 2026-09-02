"""
Email OTP Authentication Module
Provides:
1. Secure 6-digit numeric OTP generation with 5-minute expiry.
2. Email format validation.
3. SMTP email dispatch (Gmail, Outlook, Custom SMTP).
4. Zero-friction demo/local fallback when SMTP credentials are absent.
"""

import os
import re
import random
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
try:
    from dotenv import load_dotenv
    # Load environment variables
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(base_dir, ".env"))
except ImportError:
    pass

def get_config_val(key: str, default_val: str = "") -> str:
    # 1. Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key in st.secrets:
                return str(st.secrets[key])
            # Check lowercase
            if key.lower() in st.secrets:
                return str(st.secrets[key.lower()])
            # Check nested under [smtp] or [email]
            if "smtp" in st.secrets and key in st.secrets["smtp"]:
                return str(st.secrets["smtp"][key])
            if "email" in st.secrets and key in st.secrets["email"]:
                return str(st.secrets["email"][key])
    except Exception:
        pass
    
    # 2. Try OS environment
    return os.getenv(key, os.getenv(key.lower(), default_val))

class EmailAuthManager:
    """
    Manages OTP generation, email dispatch, and verification.
    """
    def __init__(self, expiry_minutes: int = 5):
        self.expiry_minutes = expiry_minutes

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """
        Validates email syntax with regex.
        """
        if not email or not isinstance(email, str):
            return False
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email.strip()))

    def generate_otp(self) -> str:
        """
        Generates a 6-digit numeric OTP.
        """
        return f"{random.randint(100000, 999999)}"

    def get_smtp_config(self) -> dict:
        """
        Dynamically fetches and cleans SMTP credentials at runtime.
        """
        server = get_config_val("SMTP_SERVER", "smtp.gmail.com").strip()
        port_str = get_config_val("SMTP_PORT", "587").strip()
        try:
            port = int(port_str)
        except ValueError:
            port = 587
        email = get_config_val("SMTP_EMAIL", "").strip()
        password = get_config_val("SMTP_PASSWORD", "").strip()
        # Clean app password: remove spaces if present (Google gives 'abcd efgh ijkl mnop')
        clean_password = password.replace(" ", "")

        return {
            "server": server,
            "port": port,
            "email": email,
            "password": clean_password,
            "is_configured": bool(email and clean_password)
        }

    def is_smtp_configured(self) -> bool:
        """
        Checks if SMTP credentials are provided.
        """
        return self.get_smtp_config()["is_configured"]

    def send_otp_email(self, recipient_email: str, recipient_name: str, otp: str) -> tuple[bool, str]:
        """
        Dispatches OTP email via SMTP.
        Returns (success: bool, message: str).
        """
        if not self.is_valid_email(recipient_email):
            return False, "Invalid email address format."

        smtp_cfg = self.get_smtp_config()

        if not smtp_cfg["is_configured"]:
            # Demo / Local fallback mode
            return True, "DEMO_MODE"

        smtp_email = smtp_cfg["email"]
        smtp_pass = smtp_cfg["password"]
        smtp_server = smtp_cfg["server"]
        smtp_port = smtp_cfg["port"]

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Your Login Verification Code: {otp} | Career Navigator"
            msg["From"] = f"Career Navigator <{smtp_email}>"
            msg["To"] = recipient_email

            html_content = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #E5E7EB; border-radius: 12px; background-color: #FFFFFF;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="color: #1E3A8A; margin-bottom: 5px;">🎓 Career Navigator</h2>
                    <p style="color: #6B7280; font-size: 14px; margin-top: 0;">AI-Powered Career Recommendation & Skill Gap Engine</p>
                </div>
                <hr style="border: 0; border-top: 1px solid #E5E7EB; margin: 20px 0;">
                <p style="font-size: 16px; color: #111827;">Hello <strong>{recipient_name}</strong>,</p>
                <p style="font-size: 15px; color: #374151;">Use the One-Time Password (OTP) below to authenticate your login:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <span style="display: inline-block; font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #2563EB; background: #EFF6FF; padding: 12px 28px; border-radius: 8px; border: 1px dashed #3B82F6;">
                        {otp}
                    </span>
                </div>
                
                <p style="font-size: 14px; color: #6B7280; text-align: center;">
                    ⏳ This code is valid for <strong>{self.expiry_minutes} minutes</strong>. Do not share this code with anyone.
                </p>
                <hr style="border: 0; border-top: 1px solid #E5E7EB; margin: 20px 0;">
                <p style="font-size: 12px; color: #9CA3AF; text-align: center;">
                    If you did not request this code, you can safely ignore this email.
                </p>
            </div>
            """

            part = MIMEText(html_content, "html")
            msg.attach(part)

            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=12)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=12)
                server.starttls()

            server.login(smtp_email, smtp_pass)
            server.sendmail(smtp_email, recipient_email, msg.as_string())
            server.quit()

            return True, "Verification email sent successfully."
        except Exception as e:
            return False, f"Email delivery error: {str(e)}"

    def verify_otp(self, entered_otp: str, actual_otp: str, timestamp: datetime) -> tuple[bool, str]:
        """
        Verifies entered OTP against expected OTP and expiration time.
        """
        if not entered_otp:
            return False, "Please enter the 6-digit OTP."

        if not actual_otp or not timestamp:
            return False, "No active OTP request found. Please request a new code."

        now = datetime.now()
        if now - timestamp > timedelta(minutes=self.expiry_minutes):
            return False, "OTP has expired. Please request a new verification code."

        if entered_otp.strip() == str(actual_otp).strip():
            return True, "OTP verified successfully."
        else:
            return False, "Invalid OTP code. Please check and try again."

auth_manager = EmailAuthManager(expiry_minutes=5)

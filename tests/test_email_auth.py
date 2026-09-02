"""
Unit tests for Email OTP Authentication Module
"""

import unittest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from email_auth import EmailAuthManager, auth_manager

class TestEmailAuth(unittest.TestCase):
    def setUp(self):
        self.auth = EmailAuthManager(expiry_minutes=5)

    def test_email_validation(self):
        self.assertTrue(self.auth.is_valid_email("student@example.com"))
        self.assertTrue(self.auth.is_valid_email("john.doe123@university.edu"))
        self.assertFalse(self.auth.is_valid_email("invalid-email"))
        self.assertFalse(self.auth.is_valid_email("@no-user.com"))
        self.assertFalse(self.auth.is_valid_email(""))

    def test_otp_generation(self):
        otp = self.auth.generate_otp()
        self.assertEqual(len(otp), 6)
        self.assertTrue(otp.isdigit())

    def test_otp_verification_success(self):
        otp = "123456"
        now = datetime.now()
        is_valid, msg = self.auth.verify_otp("123456", otp, now)
        self.assertTrue(is_valid)

    def test_otp_verification_invalid(self):
        otp = "123456"
        now = datetime.now()
        is_valid, msg = self.auth.verify_otp("654321", otp, now)
        self.assertFalse(is_valid)

    def test_otp_verification_expired(self):
        otp = "123456"
        old_time = datetime.now() - timedelta(minutes=6)
        is_valid, msg = self.auth.verify_otp("123456", otp, old_time)
        self.assertFalse(is_valid)
        self.assertIn("expired", msg.lower())

if __name__ == "__main__":
    unittest.main()

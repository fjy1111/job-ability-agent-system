import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.privacy_service import (
    candidate_privacy_label,
    hash_password,
    institution_role_access_error,
    redact_sensitive_resume_text,
    verify_password,
)


class CollaborationPrivacyTests(unittest.TestCase):
    def test_password_is_hashed_and_verifiable(self):
        encoded = hash_password("strong-password")

        self.assertNotIn("strong-password", encoded)
        self.assertLessEqual(len(encoded), 100)
        self.assertEqual(verify_password("strong-password", encoded), (True, False))
        self.assertEqual(verify_password("wrong-password", encoded), (False, False))

    def test_legacy_plaintext_password_can_be_upgraded(self):
        self.assertEqual(verify_password("old-password", "old-password"), (True, True))
        self.assertEqual(verify_password("wrong", "old-password"), (False, True))

    def test_enterprise_resume_redaction_removes_direct_identifiers(self):
        source = (
            "张三\n手机号：13812345678\n邮箱：student@example.com\n"
            "身份证：110101200001011234\n家庭住址：北京市海淀区示例路\n"
            "微信：student_wechat\n技能：Java、Spring Boot、MySQL"
        )

        redacted = redact_sensitive_resume_text(source, "张三")

        for sensitive in (
            "张三",
            "13812345678",
            "student@example.com",
            "110101200001011234",
            "北京市海淀区示例路",
            "student_wechat",
        ):
            self.assertNotIn(sensitive, redacted)
        self.assertIn("Java、Spring Boot、MySQL", redacted)

    def test_candidate_label_does_not_reveal_name(self):
        record = SimpleNamespace(
            id=7,
            student_user_id=3,
            student_name="张三",
            resume_hash="abcdef1234567890",
        )
        self.assertEqual(candidate_privacy_label(record), "候选人-ABCDEF")

    def test_institution_role_requires_server_side_access_code(self):
        with patch.dict(
            os.environ,
            {
                "SCHOOL_ROLE_ACCESS_CODE": "school-secret",
                "ENTERPRISE_ROLE_ACCESS_CODE": "enterprise-secret",
            },
        ):
            self.assertEqual(institution_role_access_error("student", ""), "")
            self.assertEqual(institution_role_access_error("school", "school-secret"), "")
            self.assertTrue(institution_role_access_error("enterprise", "wrong"))

    def test_dashboard_does_not_embed_resume_body(self):
        template = Path("app/templates/triparty_collaboration.html").read_text(encoding="utf-8")
        self.assertNotIn("application.resume_text", template)
        self.assertIn("/resume", template)

    def test_company_apply_requires_explicit_privacy_consent(self):
        template = Path("app/templates/job_match.html").read_text(encoding="utf-8")
        self.assertIn('id="resumePrivacyConsent"', template)
        self.assertIn('formData.append("privacy_consent", "accepted")', template)
        self.assertIn('formData.append("csrf_token", collaborationCsrfToken)', template)


if __name__ == "__main__":
    unittest.main()

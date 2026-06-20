import os
import unittest
from unittest.mock import patch

from app.services.model_config_service import (
    create_configured_chat_model,
    get_active_model_key,
    get_model_options,
    reset_active_model_key,
    resolve_model_config,
    set_active_model_key,
)


class ModelConfigServiceTests(unittest.TestCase):
    def test_provider_configurations_are_isolated(self):
        env = {
            "CHATGPT_API_KEY": "chatgpt-secret",
            "CHATGPT_BASE_URL": "https://chatgpt.example/v1",
            "CHATGPT_MODEL": "chatgpt-api-model",
            "GEMINI_API_KEY": "gemini-secret",
            "GEMINI_MODEL": "gemini-api-model",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "DEEPSEEK_MODEL": "deepseek-api-model",
        }
        with patch.dict(os.environ, env, clear=True):
            chatgpt = resolve_model_config("chatgpt")
            gemini = resolve_model_config("gemini")
            deepseek = resolve_model_config("deepseek")

        self.assertEqual(chatgpt["api_key"], "chatgpt-secret")
        self.assertEqual(chatgpt["model"], "chatgpt-api-model")
        self.assertEqual(gemini["api_key"], "gemini-secret")
        self.assertEqual(gemini["model"], "gemini-api-model")
        self.assertEqual(deepseek["api_key"], "deepseek-secret")
        self.assertEqual(deepseek["model"], "deepseek-api-model")

    def test_deepseek_keeps_existing_task_model_override(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "secret",
                "DEEPSEEK_MODEL": "deepseek-default",
                "ABILITY_MATCH_MODEL": "deepseek-reranker",
            },
            clear=True,
        ):
            config = resolve_model_config(
                "deepseek",
                task_name="ABILITY_MATCH",
                legacy_task_model_envs=("ABILITY_MATCH_MODEL",),
            )
        self.assertEqual(config["model"], "deepseek-reranker")

    def test_browser_options_never_include_api_keys(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "must-not-leak", "DEEPSEEK_MODEL": "deepseek-chat"},
            clear=True,
        ):
            options = get_model_options()
        serialized = repr(options)
        self.assertNotIn("must-not-leak", serialized)
        self.assertTrue(all("api_key" not in option for option in options))

    def test_request_context_selects_provider_and_resets(self):
        original = get_active_model_key()
        token = set_active_model_key("gemini")
        try:
            self.assertEqual(get_active_model_key(), "gemini")
        finally:
            reset_active_model_key(token)
        self.assertEqual(get_active_model_key(), original)

    def test_chat_client_uses_active_provider(self):
        with patch.dict(
            os.environ,
            {
                "USE_LLM": "true",
                "CHATGPT_API_KEY": "secret",
                "CHATGPT_BASE_URL": "https://chatgpt.example/v1",
                "CHATGPT_MODEL": "chatgpt-api-model",
            },
            clear=True,
        ), patch("app.services.model_config_service.ChatOpenAI") as chat_openai:
            token = set_active_model_key("chatgpt")
            try:
                create_configured_chat_model(
                    temperature=0.2,
                    timeout=30,
                    max_retries=1,
                    task_name="DIAGNOSIS",
                )
            finally:
                reset_active_model_key(token)

        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(kwargs["model"], "chatgpt-api-model")
        self.assertEqual(kwargs["api_key"], "secret")
        self.assertEqual(kwargs["base_url"], "https://chatgpt.example/v1")


if __name__ == "__main__":
    unittest.main()

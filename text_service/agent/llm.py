import json
import logging
from typing import Any, Dict, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model = genai.GenerativeModel(model_name)

    def generate_text(self, prompt: str) -> str:
        """Plain text generation with basic error handling."""
        try:
            resp = self.model.generate_content(prompt)
            return getattr(resp, "text", "").strip()
        except Exception as e:
            logger.error(f"LLM text generation error: {e}")
            return ""

    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate JSON content and parse safely; returns None on failure."""
        try:
            resp = self.model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            text = getattr(resp, "text", "").strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"LLM JSON generation error: {e}")
            return None



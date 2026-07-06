"""
=============================================================
  Face Physiognomy Project — Face Describer (VLM)
=============================================================

RESPONSIBILITY:
  - Receive 3 SectionPrompt objects from PromptBuilder
  - Receive 3 section images from FaceSectionBuilder
  - Make exactly 3 Gemini VLM calls
  - Return structured JSON per section

  No prompt building here.
  No image manipulation here.
  No geometry here.
"""

import os
import re
import json
import base64
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from prompt_builder import SectionPrompt


# =============================================================
#  Result
# =============================================================

@dataclass
class SectionDescription:
    """
    Result of one VLM call for one facial section.

    features_json : region → feature → {value, confidence}
                    Only present/visible features included.
    """
    section_id    : str
    regions       : List[str]
    features_json : Optional[Dict[str, Any]] = None
    raw_response  : str                       = ""
    success       : bool                      = False
    error         : Optional[str]             = None
    tokens_used   : int                       = 0


# =============================================================
#  Face Describer
# =============================================================

class FaceDescriber:
    """
    Makes 3 VLM calls — one per SectionPrompt + section image.

    Usage:
        describer = FaceDescriber()
        results   = describer.describe(
            prompts        = prompt_builder.build(),
            section_images = section_builder_result.sections,
        )
        # results["section_1"] → SectionDescription
    """

    MODEL         = "gemini-1.5-flash"
    MAX_RETRIES   = 3
    RETRY_DELAY_S = 5

    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai
        key = api_key or self._get_api_key()
        if not key:
            raise ValueError(
                "No Gemini API key found.\n"
                "Set GEMINI_API_KEY in environment or Kaggle/HF Secrets."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.MODEL)
        print(f"FaceDescriber ready — {self.MODEL}")

    def _get_api_key(self) -> Optional[str]:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GEMINI_API_KEY")
        except Exception:
            pass
        return None

    # ----------------------------------------------------------
    #  Image Encoding
    # ----------------------------------------------------------

    def _encode(self, img_bgr: np.ndarray) -> Dict:
        """BGR numpy → base64 JPEG dict for Gemini API."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".jpg", img_rgb,
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Image encoding failed")
        return {
            "mime_type": "image/jpeg",
            "data"     : base64.b64encode(buf.tobytes()).decode("utf-8"),
        }

    # ----------------------------------------------------------
    #  Single VLM Call
    # ----------------------------------------------------------

    def _call_once(
        self,
        prompt       : SectionPrompt,
        section_image: np.ndarray,
    ) -> SectionDescription:
        """Make one VLM call for one section."""
        try:
            img_data = self._encode(section_image)
        except Exception as e:
            return SectionDescription(
                section_id = prompt.section_id,
                regions    = prompt.regions,
                success    = False,
                error      = f"Image encoding failed: {e}",
            )

        # Gemini content: [image, prompt_text]
        contents = [
            {"mime_type": img_data["mime_type"], "data": img_data["data"]},
            prompt.prompt,
        ]

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response    = self.model.generate_content(contents=contents)
                raw_text    = response.text
                tokens_used = (
                    response.usage_metadata.total_token_count
                    if hasattr(response, "usage_metadata") else 0
                )
                features = self._parse_json(raw_text)

                return SectionDescription(
                    section_id    = prompt.section_id,
                    regions       = prompt.regions,
                    features_json = features,
                    raw_response  = raw_text,
                    success       = True,
                    tokens_used   = tokens_used,
                )

            except Exception as e:
                last_error = str(e)
                if attempt < self.MAX_RETRIES - 1:
                    print(f"      Retry {attempt+1}: {e}")
                    time.sleep(self.RETRY_DELAY_S)

        return SectionDescription(
            section_id = prompt.section_id,
            regions    = prompt.regions,
            success    = False,
            error      = f"All {self.MAX_RETRIES} attempts failed: {last_error}",
        )

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def describe(
        self,
        prompts        : Dict[str, SectionPrompt],
        section_images : Dict[str, np.ndarray],
        delay_s        : float = 2.0,
    ) -> Dict[str, SectionDescription]:
        """
        Make 3 VLM calls — one per section.

        Args:
            prompts        : output of PromptBuilder.build()
            section_images : output of FaceSectionBuilder.build().sections
            delay_s        : seconds between calls (Gemini rate limit)

        Returns:
            Dict[section_id → SectionDescription]
        """
        results = {}
        items   = [
            ("section_1", prompts["section_1"], section_images["section_1"]),
            ("section_2", prompts["section_2"], section_images["section_2"]),
            ("section_3", prompts["section_3"], section_images["section_3"]),
        ]

        for i, (section_id, prompt, image) in enumerate(items, 1):
            print(
                f"  [{i}/3] {section_id} "
                f"({', '.join(prompt.regions)})...",
                end=" ", flush=True
            )

            result = self._call_once(prompt, image)

            if result.success:
                n_regions = len(result.features_json or {})
                print(f"✓ ({n_regions} regions, {result.tokens_used} tokens)")
            else:
                print(f"✗ {result.error}")

            results[section_id] = result

            if i < len(items):
                time.sleep(delay_s)

        return results

    # ----------------------------------------------------------
    #  JSON Parser
    # ----------------------------------------------------------

    def _parse_json(self, raw_text: str) -> Dict:
        """Parse JSON from Gemini response, stripping markdown fences."""
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
        cleaned = cleaned.replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON:\n{raw_text[:300]}")

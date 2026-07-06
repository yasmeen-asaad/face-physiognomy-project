"""
=============================================================
  Face Physiognomy Project — Gradio App (V2)
=============================================================

RESPONSIBILITY:
  UI only — Gradio interface.
  Calls PhysiognomyPipeline and displays results.
  No pipeline logic here.

REQUIRED SECRETS (HF Space → Settings → Secrets):
  GEMINI_API_KEY
  GDRIVE_FOLDER_ID       (optional)
  GDRIVE_SERVICE_ACCOUNT (optional)
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline     import PhysiognomyPipeline, PipelineResult
from drive_logger import DriveLogger, SessionLog, new_session_id


# =============================================================
#  Load at Startup
# =============================================================

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "/app/rag_index")

pipeline = PhysiognomyPipeline(
    index_path     = f"{INDEX_DIR}/index.faiss",
    chunks_path    = f"{INDEX_DIR}/chunks.pkl",
    gemini_api_key = os.environ.get("GEMINI_API_KEY"),
)

drive_logger = None
try:
    drive_logger = DriveLogger()
    print("Drive logger ready.")
except Exception as e:
    print(f"Drive logging disabled: {e}")


# =============================================================
#  Logging Helper
# =============================================================

def _log(session: SessionLog, original_img: Optional[np.ndarray] = None):
    if drive_logger is None:
        return
    try:
        drive_logger.log_session(session, original_img=original_img)
    except Exception as e:
        print(f"Logging error (non-fatal): {e}")


# =============================================================
#  Main Analysis Function
# =============================================================

def analyze(
    front_image   : Optional[np.ndarray],
    profile_image : Optional[np.ndarray],
    profile_side  : str,
) -> tuple:
    """
    Called by Gradio on button click.
    Returns (status_text, report_text).
    """
    session_id = new_session_id()
    start      = time.time()

    session = SessionLog(
        session_id  = session_id,
        timestamp   = datetime.now(timezone.utc).isoformat(),
        status      = "started",
        latency_sec = 0.0,
    )

    if front_image is None:
        return "⚠ Please upload a front face image.", ""

    # Gradio sends RGB — convert to BGR
    front_bgr    = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
    profile_bgr  = None
    profile_side_ = None

    if profile_image is not None:
        profile_bgr   = cv2.cvtColor(profile_image, cv2.COLOR_RGB2BGR)
        profile_side_ = profile_side.lower() if profile_side else None

    result: PipelineResult = pipeline.run(
        front_image   = front_bgr,
        profile_image = profile_bgr,
        profile_side  = profile_side_,
    )

    latency = round(time.time() - start, 2)

    if result.success:
        profile_note = (
            f"Profile: {profile_side_} ✓"
            if result.has_profile else "Profile: not provided"
        )
        status = (
            f"✓ Analysis complete in {latency}s\n"
            f"Session  : {session_id}\n"
            f"{profile_note}\n"
            f"Regions  : {', '.join(result.visual_features.keys())}"
        )
        session.status      = "success"
        session.report_text = result.report
    else:
        status            = f"⚠ {result.error}"
        session.status    = result.detection_status or "error"
        session.error_msg = result.error

    session.latency_sec = latency
    _log(session, original_img=front_bgr)

    return status, result.report if result.success else ""


# =============================================================
#  Gradio UI
# =============================================================

def build_ui():
    with gr.Blocks(title="Face Physiognomy Analyzer") as demo:

        gr.Markdown("""
# 🔍 Face Physiognomy Analyzer
Analyze facial features based on the principles of physiognomy.

**Requirements:**
- Front image: clear, neutral expression, good lighting
- Profile image (optional): side view for deeper feature analysis
        """)

        with gr.Row():

            with gr.Column(scale=1):
                front_input = gr.Image(
                    label  = "Front Face Image (required)",
                    type   = "numpy",
                    height = 320,
                )
                profile_input = gr.Image(
                    label  = "Profile Face Image (optional)",
                    type   = "numpy",
                    height = 320,
                )
                side_input = gr.Radio(
                    choices = ["Left", "Right"],
                    value   = "Left",
                    label   = "Profile Side",
                    info    = "Which side is shown in the profile image?",
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Face",
                    variant = "primary",
                    size    = "lg",
                )

            with gr.Column(scale=1):
                status_out = gr.Textbox(
                    label       = "Status",
                    lines       = 5,
                    interactive = False,
                )
                report_out = gr.Textbox(
                    label       = "Physiognomy Report",
                    lines       = 24,
                    interactive = False,
                )

        gr.Markdown("""
---
*For educational and entertainment purposes only.*
*Based on "Amazing Face Reading" by Mac Fulfer.*
        """)

        analyze_btn.click(
            fn      = analyze,
            inputs  = [front_input, profile_input, side_input],
            outputs = [status_out, report_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()

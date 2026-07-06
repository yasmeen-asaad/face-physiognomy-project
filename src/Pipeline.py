"""
=============================================================
  Face Physiognomy Project — Pipeline Orchestrator (V2)
=============================================================

RESPONSIBILITY:
  Run the full V2 pipeline end-to-end.
  Calls each module in the correct order.
  Passes outputs between modules.
  Contains no business logic of its own.

PIPELINE:
  Front Image + optional Profile Image
        │
        ├──────────────────────────┐
        ▼                          ▼
  FrontImageProcessor      ProfileImageProcessor
  (face_crop + landmarks)  (profile_crop)
        │                          │
        ├──────────────────────────┤
        │                          │
        ▼                          ▼
  GeometryExtractor      FaceSectionBuilder
  (measurements)         (3 combined section images)
        │                          │
        └────────────┬─────────────┘
                     ▼
             PromptBuilder
          (3 SectionPrompt objects)
                     │
                     ▼
             FaceDescriber
          (3 Gemini VLM calls)
                     │
                     ▼
       Visual Feature JSON
       (false features removed)
                     │
                     ▼
             RAGRetriever
                     │
                     ▼
           ReportGenerator

USAGE:
    from pipeline import PhysiognomyPipeline

    pipeline = PhysiognomyPipeline(
        index_path  = "/path/to/index.faiss",
        chunks_path = "/path/to/chunks.pkl",
    )
    result = pipeline.run(
        front_image   = front_bgr,
        profile_image = profile_bgr,  # or None
        profile_side  = "left",       # or "right" or None
    )
    if result.success:
        print(result.report)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

from front_image_processor   import FrontImageProcessor
from profile_image_processor import ProfileImageProcessor
from geometry_extractor      import GeometryExtractor
from face_section_builder    import FaceSectionBuilder
from prompt_builder          import PromptBuilder
from face_describer          import FaceDescriber
from rag_retriever           import PhysiognomyRetriever
from report_generator        import ReportGenerator


# =============================================================
#  Pipeline Result
# =============================================================

@dataclass
class PipelineResult:
    """
    Complete output of one pipeline run.

    Primary fields:
        success         : True if pipeline completed without fatal error
        report          : final physiognomy report text
        error           : error message if success=False

    Debug / logging fields:
        visual_features : merged JSON from all 3 VLM calls
        rag_evidence    : retrieved book passages per region
        face_crop       : cropped front face image
        section_images  : the 3 combined images sent to VLM
        measurements    : geometry measurements from landmarks
        has_profile     : whether a profile image was used
        latency_sec     : total wall-clock time
        detection_status: face detection status string
    """
    success          : bool
    report           : str                   = ""
    error            : Optional[str]         = None
    visual_features  : Dict[str, Any]        = field(default_factory=dict)
    rag_evidence     : Dict[str, Any]        = field(default_factory=dict)
    face_crop        : Optional[np.ndarray]  = None
    section_images   : Dict[str, Any]        = field(default_factory=dict)
    measurements     : Dict[str, Any]        = field(default_factory=dict)
    has_profile      : bool                  = False
    latency_sec      : float                 = 0.0
    detection_status : str                   = ""


# =============================================================
#  Pipeline
# =============================================================

class PhysiognomyPipeline:
    """
    Full V2 physiognomy analysis pipeline.

    Instantiate once — all models loaded at init time.
    Call run() for each image pair.
    """

    def __init__(
        self,
        index_path     : str,
        chunks_path    : str,
        gemini_api_key : Optional[str] = None,
        vlm_delay_s    : float         = 2.0,
    ):
        self.vlm_delay_s = vlm_delay_s

        print("Loading pipeline...")

        self.front_processor   = FrontImageProcessor()
        self.profile_processor = ProfileImageProcessor()
        print("  ✓ Image processors")

        self.retriever = PhysiognomyRetriever(
            index_path  = index_path,
            chunks_path = chunks_path,
        )
        print("  ✓ RAG retriever")

        self.describer = FaceDescriber(api_key=gemini_api_key)
        print("  ✓ VLM describer")

        self.generator = ReportGenerator(api_key=gemini_api_key)
        print("  ✓ Report generator")

        print("Pipeline ready.\n")

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def run(
        self,
        front_image   : np.ndarray,
        profile_image : Optional[np.ndarray] = None,
        profile_side  : Optional[str]        = None,
    ) -> PipelineResult:
        """
        Run full pipeline on front (+ optional profile) image.

        Args:
            front_image   : BGR numpy array
            profile_image : BGR numpy array or None
            profile_side  : "left" | "right" | None — from user input

        Returns:
            PipelineResult
        """
        start = time.time()

        # ── Step 1a: Front face detection ─────────────────────
        print("[1] Face detection...")
        front_result = self.front_processor.process(front_image)

        if not front_result.is_valid:
            return PipelineResult(
                success          = False,
                error            = front_result.message,
                detection_status = front_result.status.value,
                latency_sec      = round(time.time() - start, 2),
            )
        print(f"  ✓ Front face detected")

        # ── Step 1b: Profile face detection (optional) ─────────
        profile_crop = None
        if profile_image is not None:
            print("[1b] Profile detection...")
            profile_result = self.profile_processor.process(
                profile_image, side=profile_side
            )
            if profile_result.is_valid:
                profile_crop = profile_result.profile_crop
                print(f"  ✓ Profile face detected ({profile_side})")
            else:
                print(f"  ⚠ Profile detection failed: {profile_result.message}")
                print("    Continuing with front image only.")

        has_profile = profile_crop is not None

        # ── Step 2: Geometry extraction ───────────────────────
        # Runs independently of section building.
        print("[2] Computing geometry...")
        measurements = GeometryExtractor(
            crop_landmarks=front_result.crop_landmarks
        ).compute()
        print("  ✓ Measurements computed")

        # ── Step 3: Section building ──────────────────────────
        # Runs independently of geometry extraction.
        print("[3] Building sections...")
        section_builder = FaceSectionBuilder(
            face_crop      = front_result.face_crop,
            crop_landmarks = front_result.crop_landmarks,
            profile_crop   = profile_crop,
            profile_side   = profile_side,
        )
        sections = section_builder.build()
        print(f"  ✓ 3 sections built (profile: {has_profile})")

        # ── Step 4: Prompt building ───────────────────────────
        print("[4] Building prompts...")
        prompts = PromptBuilder(
            measurements = measurements,
            has_profile  = sections.has_profile,
            profile_side = sections.profile_side,
        ).build()
        print("  ✓ 3 prompts ready")

        # ── Step 5: VLM description ───────────────────────────
        print("[5] VLM analysis (3 calls)...")
        descriptions = self.describer.describe(
            prompts        = prompts,
            section_images = sections.sections,
            delay_s        = self.vlm_delay_s,
        )

        # Check at least one section succeeded
        if not any(d.success for d in descriptions.values()):
            return PipelineResult(
                success          = False,
                error            = "All 3 VLM calls failed.",
                face_crop        = front_result.face_crop,
                section_images   = sections.sections,
                measurements     = measurements,
                has_profile      = has_profile,
                detection_status = front_result.status.value,
                latency_sec      = round(time.time() - start, 2),
            )

        # Merge JSON from all sections into one dict
        # Section 2 "eyes" overwrites Section 1 "eyes" if both present
        # (Section 2 has more eye detail — white showing, etc.)
        visual_features: Dict[str, Any] = {}
        for section_id in ["section_1", "section_2", "section_3"]:
            desc = descriptions.get(section_id)
            if desc and desc.success and desc.features_json:
                # Deep merge: don't overwrite, extend
                for region, features in desc.features_json.items():
                    if region not in visual_features:
                        visual_features[region] = {}
                    if isinstance(features, dict):
                        visual_features[region].update(features)
                    else:
                        visual_features[region] = features

        print(f"  ✓ Visual features: {list(visual_features.keys())}")

        # ── Step 6: RAG retrieval ─────────────────────────────
        print("[6] RAG retrieval...")
        rag_evidence = self.retriever.search_all_parts(
            descriptions, top_k=3
        )
        print(f"  ✓ Retrieved evidence for {len(rag_evidence)} regions")

        # ── Step 7: Report generation ─────────────────────────
        print("[7] Generating report...")
        report = self.generator.generate(rag_evidence)

        latency = round(time.time() - start, 2)

        if not report.success:
            return PipelineResult(
                success          = False,
                error            = f"Report failed: {report.error}",
                visual_features  = visual_features,
                rag_evidence     = rag_evidence,
                face_crop        = front_result.face_crop,
                section_images   = sections.sections,
                measurements     = measurements,
                has_profile      = has_profile,
                detection_status = front_result.status.value,
                latency_sec      = latency,
            )

        print(f"  ✓ Report generated ({latency}s total)")

        return PipelineResult(
            success          = True,
            report           = report.report_text,
            visual_features  = visual_features,
            rag_evidence     = rag_evidence,
            face_crop        = front_result.face_crop,
            section_images   = sections.sections,
            measurements     = measurements,
            has_profile      = has_profile,
            detection_status = front_result.status.value,
            latency_sec      = latency,
        )

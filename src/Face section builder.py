"""
=============================================================
  Face Physiognomy Project — Face Section Builder
=============================================================

RESPONSIBILITY:
  - Split front face crop into 3 sections (using landmarks)
  - Split profile crop into 3 sections (proportional — no landmarks)
  - Merge each front section with its profile counterpart
  - Separate merged sections with a white line
  - Return 3 combined images

  No landmark calculations.
  No geometry measurements.
  No prompt logic.

SECTION BOUNDARIES (front — landmark-based):

  Y axis:
    Section 1 top    : landmark[10].y  − 40% face_height  (clamped to 0)
    Section 1 bottom : landmark[168].y (nose bridge)

    Section 2 top    : min(landmark[70].y, landmark[300].y) (eyebrow top)
    Section 2 bottom : landmark[17].y  (mouth bottom)

    Section 3 top    : landmark[4].y   (nose tip)
    Section 3 bottom : landmark[152].y + 10% face_height

  X axis (same for all sections):
    x_left  : landmark[234].x − 20% face_width  (clamped to 0)
    x_right : landmark[454].x + 20% face_width  (clamped to crop width)

PROFILE SECTIONS:
  Profile has no landmarks — proportional split using same y-ratios
  as the front image sections.

MERGE:
  Each combined image = vstack([front_section, WHITE_LINE, profile_section])
  White line = 6px — neutral, no visual interference with face colors.
  If no profile: combined = front_section only.

  All images resized to TARGET_WIDTH=480 before merging
  to ensure consistent dimensions.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


TARGET_WIDTH       = 480
WHITE_LINE_HEIGHT  = 6


# =============================================================
#  Helpers
# =============================================================

def _lm(landmarks: List[Tuple[int,int]], idx: int) -> Tuple[int,int]:
    return landmarks[idx] if idx < len(landmarks) else (0, 0)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _resize_w(img: np.ndarray, width: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if w == 0:
        return np.zeros((1, width, 3), dtype=np.uint8)
    return cv2.resize(img, (width, max(1, int(h * width / w))))


def _white_line(width: int) -> np.ndarray:
    line = np.full((WHITE_LINE_HEIGHT, width, 3), 255, dtype=np.uint8)
    return line


def _safe_crop(img: np.ndarray, y1: int, y2: int,
               x1: int, x2: int) -> np.ndarray:
    if y2 <= y1 or x2 <= x1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


# =============================================================
#  Boundary Calculator
# =============================================================

def _compute_bounds(
    landmarks : List[Tuple[int,int]],
    crop_h    : int,
    crop_w    : int,
) -> Dict:
    """
    Compute pixel boundaries for 3 sections from landmarks.
    Returns dict with all boundary values clamped to image dimensions.
    """
    lm10  = _lm(landmarks, 10)    # forehead top
    lm152 = _lm(landmarks, 152)   # chin bottom
    lm234 = _lm(landmarks, 234)   # left jaw hinge
    lm454 = _lm(landmarks, 454)   # right jaw hinge
    lm168 = _lm(landmarks, 168)   # nose bridge
    lm70  = _lm(landmarks, 70)    # right eyebrow top
    lm300 = _lm(landmarks, 300)   # left eyebrow top
    lm17  = _lm(landmarks, 17)    # mouth bottom
    lm4   = _lm(landmarks, 4)     # nose tip

    fh = max(lm152[1] - lm10[1], 1)
    fw = max(lm454[0] - lm234[0], 1)

    # X boundaries — same for all sections
    x1 = _clamp(lm234[0] - int(fw * 0.20), 0, crop_w - 1)
    x2 = _clamp(lm454[0] + int(fw * 0.20), 0, crop_w - 1)

    # Section 1: forehead → nose bridge
    s1_y1 = _clamp(lm10[1] - int(fh * 0.40), 0, crop_h - 1)
    s1_y2 = _clamp(lm168[1],                  0, crop_h - 1)

    # Section 2: eyebrows → mouth bottom
    s2_y1 = _clamp(min(lm70[1], lm300[1]), 0, crop_h - 1)
    s2_y2 = _clamp(lm17[1],               0, crop_h - 1)

    # Section 3: nose tip → chin (+ 10% below)
    s3_y1 = _clamp(lm4[1],                         0, crop_h - 1)
    s3_y2 = _clamp(lm152[1] + int(fh * 0.10), 0, crop_h - 1)

    return {
        "x1": x1, "x2": x2,
        "s1": (s1_y1, s1_y2),
        "s2": (s2_y1, s2_y2),
        "s3": (s3_y1, s3_y2),
        "fh": fh,
    }


# =============================================================
#  Profile Splitter
# =============================================================

def _split_profile(
    profile_crop : np.ndarray,
    bounds       : Dict,
    crop_h       : int,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Split profile crop into 3 sections using the same y-proportions
    as the front image (no landmarks available for profile).

    Profile is resized to match front crop height first.
    """
    if profile_crop is None:
        return {"s1": None, "s2": None, "s3": None}

    ph, pw = profile_crop.shape[:2]
    if ph == 0:
        return {"s1": None, "s2": None, "s3": None}

    # Resize profile to same height as front crop
    scale   = crop_h / ph
    resized = cv2.resize(profile_crop, (max(1, int(pw * scale)), crop_h))
    rh      = resized.shape[0]

    def crop(y1, y2):
        y1 = _clamp(y1, 0, rh - 1)
        y2 = _clamp(y2, 0, rh)
        if y2 <= y1:
            return None
        return resized[y1:y2, :].copy()

    return {
        "s1": crop(*bounds["s1"]),
        "s2": crop(*bounds["s2"]),
        "s3": crop(*bounds["s3"]),
    }


# =============================================================
#  Merge
# =============================================================

def _merge(
    front   : np.ndarray,
    profile : Optional[np.ndarray],
) -> np.ndarray:
    """
    Stack front (top) + white line + profile (bottom).
    If no profile: return front only.
    Both resized to TARGET_WIDTH before merging.
    """
    front_r = _resize_w(front, TARGET_WIDTH)

    if profile is None or profile.size == 0:
        return front_r

    profile_r = _resize_w(profile, TARGET_WIDTH)
    separator = _white_line(TARGET_WIDTH)

    return np.vstack([front_r, separator, profile_r])


# =============================================================
#  Result
# =============================================================

@dataclass
class SectionBuilderResult:
    """
    3 combined section images ready for VLM.

    sections:
      "section_1" : upper  (forehead + eyes + nose bridge)
      "section_2" : middle (eyebrows + nose + cheeks)
      "section_3" : lower  (mouth + jaw + chin)

    has_profile  : whether profile was merged into sections
    profile_side : "left", "right", or None
    """
    sections     : Dict[str, np.ndarray]
    has_profile  : bool
    profile_side : Optional[str]


# =============================================================
#  Main Class
# =============================================================

class FaceSectionBuilder:
    """
    Splits front + optional profile into 3 combined section images.

    Usage:
        builder = FaceSectionBuilder(
            face_crop      = front_result.face_crop,
            crop_landmarks = front_result.crop_landmarks,
            profile_crop   = profile_result.profile_crop,  # or None
            profile_side   = profile_result.side,          # or None
        )
        result = builder.build()
        # result.sections["section_1"] → upper face image
    """

    def __init__(
        self,
        face_crop      : np.ndarray,
        crop_landmarks : List[Tuple[int,int]],
        profile_crop   : Optional[np.ndarray] = None,
        profile_side   : Optional[str]        = None,
    ):
        self.face_crop      = face_crop
        self.crop_landmarks = crop_landmarks
        self.profile_crop   = profile_crop
        self.profile_side   = profile_side
        self.crop_h, self.crop_w = face_crop.shape[:2]

    def build(self) -> SectionBuilderResult:
        """
        Pipeline:
          1. Compute section boundaries from landmarks
          2. Crop 3 front sections
          3. Split profile into 3 proportional sections
          4. Merge each pair with white separator line
        """
        # Step 1
        bounds = _compute_bounds(
            self.crop_landmarks, self.crop_h, self.crop_w
        )
        x1, x2 = bounds["x1"], bounds["x2"]

        # Step 2 — front sections
        f1 = _safe_crop(self.face_crop, *bounds["s1"], x1, x2)
        f2 = _safe_crop(self.face_crop, *bounds["s2"], x1, x2)
        f3 = _safe_crop(self.face_crop, *bounds["s3"], x1, x2)

        # Step 3 — profile sections
        p  = _split_profile(self.profile_crop, bounds, self.crop_h)

        # Step 4 — merge
        return SectionBuilderResult(
            sections={
                "section_1": _merge(f1, p["s1"]),
                "section_2": _merge(f2, p["s2"]),
                "section_3": _merge(f3, p["s3"]),
            },
            has_profile  = self.profile_crop is not None,
            profile_side = self.profile_side,
        )

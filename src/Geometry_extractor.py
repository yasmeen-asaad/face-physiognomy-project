"""
=============================================================
  Face Physiognomy Project — Geometry Extractor
=============================================================

RESPONSIBILITY:
  Compute geometric measurements from MediaPipe face landmarks.

INPUT:
  crop_landmarks : List[Tuple[int, int]]
    - 468 (x, y) pixel coordinates in face_crop space
    - Comes directly from FaceDetectorValidator.result.crop_landmarks

OUTPUT:
  Dict with 3 keys — one per face section:
  {
    "section_1": { eye_spacing, eye_angle, forehead_ratio, iris_ratio, ... },
    "section_2": { nose_width, nose_length, eye_white_showing, ... },
    "section_3": { mouth_width, mouth_angle, jaw_width, chin_length, ... },
  }

  Values are either:
    - float ratios (relative to face_height or face_width)
    - string labels (wide/average/close, upward/level/downward, ...)
    - bool flags (nose_large, chin_long, ...)

  WHY ratios instead of pixel values?
    Ratios are scale-invariant — they work the same for a 200px
    face and a 2000px face. The VLM and RAG don't care about
    absolute pixel size, they care about proportions.

USAGE:
    from geometry_extractor import GeometryExtractor

    extractor    = GeometryExtractor(crop_landmarks)
    measurements = extractor.compute()

    # measurements["section_1"]["eye_spacing_label"] → "wide"
    # measurements["section_2"]["nose_width_label"]  → "thin"
    # measurements["section_3"]["mouth_angle"]       → "turns_down"
"""

import numpy as np
from typing import List, Tuple, Dict


# =============================================================
#  Landmark Index Reference
# =============================================================
#
#  Only the indices we actually use for measurements.
#  Full reference: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

class LMIdx:
    # Forehead / face height reference
    FOREHEAD_TOP   = 10
    CHIN_BOTTOM    = 152

    # Face width reference (jaw hinges)
    JAW_LEFT       = 234
    JAW_RIGHT      = 454

    # Nose
    NOSE_BRIDGE    = 168   # top of nose bridge (between eyes)
    NOSE_TIP       = 4
    NOSTRIL_LEFT   = 129
    NOSTRIL_RIGHT  = 358

    # Eyes — right eye (viewer's perspective)
    R_EYE_TOP      = 159
    R_EYE_BOTTOM   = 145
    R_EYE_INNER    = 133   # inner corner (towards nose)
    R_EYE_OUTER    = 33    # outer corner (towards ear)

    # Eyes — left eye (viewer's perspective)
    L_EYE_TOP      = 386
    L_EYE_BOTTOM   = 374
    L_EYE_INNER    = 362
    L_EYE_OUTER    = 263

    # Eyebrows
    R_BROW_TOP     = 70
    L_BROW_TOP     = 300

    # Mouth
    MOUTH_LEFT     = 61
    MOUTH_RIGHT    = 291
    MOUTH_CENTER   = 13    # upper lip center (for angle calculation)
    MOUTH_BOTTOM   = 17    # lower lip / chin boundary

    # Chin width
    CHIN_LEFT      = 136
    CHIN_RIGHT     = 365


# =============================================================
#  Geometry Extractor
# =============================================================

class GeometryExtractor:
    """
    Computes all geometric measurements from face landmarks.

    Organized into 3 sections matching face_section_builder.py:
      section_1 → forehead, eyes (spacing, angle, iris, white)
      section_2 → nose (width, length, size), eye white showing
      section_3 → mouth (size, angle), jaw, chin, facial dominance

    All measurements use ratios relative to face_height or face_width
    so they are scale-invariant across different image resolutions.
    """

    def __init__(self, crop_landmarks: List[Tuple[int, int]]):
        """
        Args:
            crop_landmarks : List of (x, y) pixel tuples in crop space.
                             Index N corresponds to MediaPipe landmark N.
                             Comes from FaceDetectorValidator.result.crop_landmarks
        """
        self.lm = crop_landmarks

    # ----------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------

    def _pt(self, idx: int) -> Tuple[int, int]:
        """Get landmark point safely. Returns (0, 0) if out of range."""
        if idx < len(self.lm):
            return self.lm[idx]
        return (0, 0)

    def _dist(self, idx1: int, idx2: int) -> float:
        """Euclidean distance between two landmarks."""
        p1 = np.array(self._pt(idx1))
        p2 = np.array(self._pt(idx2))
        return float(np.linalg.norm(p1 - p2))

    def _ratio(self, numerator: float, denominator: float,
               decimals: int = 2) -> float:
        """Safe division with rounding."""
        if denominator <= 0:
            return 0.0
        return round(numerator / denominator, decimals)

    # ----------------------------------------------------------
    #  Reference Measurements
    # ----------------------------------------------------------

    def _face_height(self) -> float:
        """Vertical distance from forehead top to chin bottom."""
        return max(self._dist(LMIdx.FOREHEAD_TOP, LMIdx.CHIN_BOTTOM), 1.0)

    def _face_width(self) -> float:
        """Horizontal distance between jaw hinges."""
        return max(self._dist(LMIdx.JAW_LEFT, LMIdx.JAW_RIGHT), 1.0)

    # ----------------------------------------------------------
    #  Section 1: Forehead + Eyes
    # ----------------------------------------------------------

    def _eye_spacing(self, fh: float, fw: float) -> Dict:
        """
        Eye spacing = distance between inner eye corners / width of one eye.

        Book rule: if gap between eyes ≈ one eye width → average spacing.
        Wider gap → wide-set. Narrower → close-set.
        """
        between_eyes = self._dist(LMIdx.R_EYE_INNER, LMIdx.L_EYE_INNER)
        eye_width    = self._dist(LMIdx.R_EYE_INNER, LMIdx.R_EYE_OUTER)
        ratio        = self._ratio(between_eyes, eye_width)

        if ratio > 1.15:   label = "wide"
        elif ratio < 0.85: label = "close"
        else:              label = "average"

        return {"eye_spacing_ratio": ratio, "eye_spacing_label": label}

    def _eye_angle(self) -> Dict:
        """
        Eye angle per eye: compare y of outer corner vs inner corner.

        Book rule: draw a line from inner to outer corner.
          outer higher than inner → upward angle
          outer lower  than inner → downward angle
          same level              → level

        Note: in image coordinates, y increases downward.
          So outer_y < inner_y means outer is HIGHER → upward.
        """
        def angle(inner_idx, outer_idx) -> str:
            inner_y = self._pt(inner_idx)[1]
            outer_y = self._pt(outer_idx)[1]
            dy = outer_y - inner_y   # negative = outer is higher
            if dy < -4:   return "upward"
            if dy >  4:   return "downward"
            return "level"

        return {
            "right_eye_angle": angle(LMIdx.R_EYE_INNER, LMIdx.R_EYE_OUTER),
            "left_eye_angle" : angle(LMIdx.L_EYE_INNER, LMIdx.L_EYE_OUTER),
        }

    def _iris_size(self) -> Dict:
        """
        Iris size ≈ eye opening height / eye width.

        Book rule: large iris fills the eye, small iris has visible
        white on sides. We approximate using right eye landmarks.
        """
        eye_h  = self._dist(LMIdx.R_EYE_TOP,   LMIdx.R_EYE_BOTTOM)
        eye_w  = self._dist(LMIdx.R_EYE_INNER, LMIdx.R_EYE_OUTER)
        ratio  = self._ratio(eye_h, eye_w)
        label  = "large" if ratio > 0.35 else "small"
        return {"iris_size_ratio": ratio, "iris_size_label": label}

    def _forehead(self, fh: float) -> Dict:
        """
        Forehead height = distance from forehead top landmark to
        top of eyebrows, normalized by face height.
        """
        brow_y      = min(self._pt(LMIdx.R_BROW_TOP)[1],
                          self._pt(LMIdx.L_BROW_TOP)[1])
        forehead_h  = max(brow_y - self._pt(LMIdx.FOREHEAD_TOP)[1], 0)
        ratio       = self._ratio(forehead_h, fh)

        if ratio > 0.35:   size = "high"
        elif ratio < 0.22: size = "low"
        else:              size = "medium"

        return {"forehead_height_ratio": ratio, "forehead_size": size}

    def section_1(self) -> Dict:
        """All measurements for Section 1 (forehead + eyes)."""
        fh = self._face_height()
        fw = self._face_width()
        return {
            **self._eye_spacing(fh, fw),
            **self._eye_angle(),
            **self._iris_size(),
            **self._forehead(fh),
        }

    # ----------------------------------------------------------
    #  Section 2: Nose + Eye White
    # ----------------------------------------------------------

    def _nose(self, fh: float, fw: float) -> Dict:
        """
        Nose width  = distance between nostril outer edges / face width.
        Nose length = distance from bridge to tip / face height.

        Book rule:
          Wide nose  → nose_width_ratio > 0.38
          Thin nose  → nose_width_ratio < 0.25
          Long nose  → nose_length_ratio > 0.38
          Short nose → nose_length_ratio < 0.25
        """
        nose_w = self._dist(LMIdx.NOSTRIL_LEFT, LMIdx.NOSTRIL_RIGHT)
        nose_l = self._dist(LMIdx.NOSE_BRIDGE,  LMIdx.NOSE_TIP)

        w_ratio = self._ratio(nose_w, fw)
        l_ratio = self._ratio(nose_l, fh)

        if w_ratio > 0.38:   w_label = "wide"
        elif w_ratio < 0.25: w_label = "thin"
        else:                w_label = "average"

        if l_ratio > 0.38:   l_label = "long"
        elif l_ratio < 0.25: l_label = "short"
        else:                l_label = "average"

        return {
            "nose_width_ratio" : w_ratio,
            "nose_width_label" : w_label,
            "nose_length_ratio": l_ratio,
            "nose_length_label": l_label,
            "nose_large"       : w_ratio > 0.38 or l_ratio > 0.38,
        }

    def _eye_white_showing(self) -> Dict:
        """
        Eye white showing = where is the iris relative to the eye center.

        Book rule:
          iris below center → white shows above iris
          iris above center → white shows below iris
          centered          → moderate (normal)

        We use right eye top/bottom as vertical reference.
        """
        eye_top_y    = self._pt(LMIdx.R_EYE_TOP)[1]
        eye_bot_y    = self._pt(LMIdx.R_EYE_BOTTOM)[1]
        eye_center_y = (eye_top_y + eye_bot_y) / 2

        # Approximate iris center as midpoint of top landmark
        # (MediaPipe doesn't give us iris center without refine_landmarks)
        iris_y = self._pt(LMIdx.R_EYE_TOP)[1]
        dy     = iris_y - eye_center_y

        if dy > 4:    label = "white_above_iris"
        elif dy < -4: label = "white_below_iris"
        else:         label = "moderate"

        return {"eye_white_showing": label}

    def section_2(self) -> Dict:
        """All measurements for Section 2 (nose + eye white)."""
        fh = self._face_height()
        fw = self._face_width()
        return {
            **self._nose(fh, fw),
            **self._eye_white_showing(),
        }

    # ----------------------------------------------------------
    #  Section 3: Mouth + Jaw + Chin + Facial Dominance
    # ----------------------------------------------------------

    def _mouth(self, fw: float) -> Dict:
        """
        Mouth size  = mouth width / face width.
        Mouth angle = center of mouth vs corners (book: connect 3 dots).

        Book rule for angle:
          Draw a dot at each mouth corner and one in the center.
          Connect them — if center is higher than corners → turns up.
        """
        mouth_w = self._dist(LMIdx.MOUTH_LEFT, LMIdx.MOUTH_RIGHT)
        w_ratio = self._ratio(mouth_w, fw)

        if w_ratio > 0.48:   size = "large"
        elif w_ratio < 0.32: size = "small"
        else:                size = "average"

        # Angle: center y vs average of corners y
        # In image coords: lower y = higher in image
        center_y  = self._pt(LMIdx.MOUTH_CENTER)[1]
        corners_y = (self._pt(LMIdx.MOUTH_LEFT)[1] +
                     self._pt(LMIdx.MOUTH_RIGHT)[1]) / 2
        dy = center_y - corners_y

        if dy < -4:   angle = "turns_up"
        elif dy > 4:  angle = "turns_down"
        else:         angle = "straight"

        return {
            "mouth_width_ratio": w_ratio,
            "mouth_size_label" : size,
            "mouth_angle"      : angle,
        }

    def _jaw(self, fw: float) -> Dict:
        """
        Jaw width = distance between jaw hinges / face width.
        (Ratio > 1.0 because jaw hinges define face_width itself —
         we use a slightly wider reference for comparison.)
        """
        jaw_w   = self._dist(LMIdx.JAW_LEFT, LMIdx.JAW_RIGHT)
        ratio   = self._ratio(jaw_w, fw)
        label   = "wide" if ratio > 0.95 else ("narrow" if ratio < 0.75 else "average")
        return {"jaw_width_ratio": ratio, "jaw_size_label": label}

    def _chin(self, fh: float, fw: float) -> Dict:
        """
        Chin length = distance from mouth bottom to chin tip / face height.
        Chin width  = distance between outer chin edges / face width.
        """
        chin_l   = self._dist(LMIdx.MOUTH_BOTTOM, LMIdx.CHIN_BOTTOM)
        chin_w   = self._dist(LMIdx.CHIN_LEFT,    LMIdx.CHIN_RIGHT)
        l_ratio  = self._ratio(chin_l, fh)
        w_ratio  = self._ratio(chin_w, fw)

        if w_ratio > 0.55:   w_label = "broad"
        elif w_ratio < 0.35: w_label = "small"
        else:                w_label = "average"

        return {
            "chin_length_ratio": l_ratio,
            "chin_long"        : l_ratio > 0.22,
            "chin_width_ratio" : w_ratio,
            "chin_width_label" : w_label,
        }

    def _facial_dominance(self) -> Dict:
        """
        Divide face into 3 horizontal sections and find which is largest.

        Book rule:
          Top    = hairline (lm 10) to eyebrows
          Middle = eyebrows to nose tip
          Bottom = nose tip to chin

        dominant  → largest section
        recessive → smallest section
        """
        top_y    = self._pt(LMIdx.FOREHEAD_TOP)[1]
        brow_y   = min(self._pt(LMIdx.R_BROW_TOP)[1],
                       self._pt(LMIdx.L_BROW_TOP)[1])
        tip_y    = self._pt(LMIdx.NOSE_TIP)[1]
        chin_y   = self._pt(LMIdx.CHIN_BOTTOM)[1]

        top_h    = max(brow_y - top_y,  1)
        mid_h    = max(tip_y  - brow_y, 1)
        bot_h    = max(chin_y - tip_y,  1)
        total_h  = top_h + mid_h + bot_h

        top_r    = self._ratio(top_h, total_h)
        mid_r    = self._ratio(mid_h, total_h)
        bot_r    = self._ratio(bot_h, total_h)

        heights  = {"top": top_h, "middle": mid_h, "bottom": bot_h}
        dominant  = max(heights, key=heights.get)
        recessive = min(heights, key=heights.get)

        return {
            "facial_sections": {
                "top_ratio"   : top_r,
                "middle_ratio": mid_r,
                "bottom_ratio": bot_r,
                "dominant"    : dominant,
                "recessive"   : recessive,
            }
        }

    def section_3(self) -> Dict:
        """All measurements for Section 3 (mouth + jaw + chin + dominance)."""
        fh = self._face_height()
        fw = self._face_width()
        return {
            **self._mouth(fw),
            **self._jaw(fw),
            **self._chin(fh, fw),
            **self._facial_dominance(),
        }

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def compute(self) -> Dict[str, Dict]:
        """
        Run all measurements and return structured dict.

        Returns:
            {
              "section_1": { eye_spacing, eye_angle, iris_size, forehead },
              "section_2": { nose_width, nose_length, eye_white_showing },
              "section_3": { mouth, jaw, chin, facial_dominance },
            }
        """
        return {
            "section_1": self.section_1(),
            "section_2": self.section_2(),
            "section_3": self.section_3(),
        }

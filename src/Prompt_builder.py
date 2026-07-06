"""
=============================================================
  Face Physiognomy Project — Prompt Builder
=============================================================

RESPONSIBILITY:
  Build 3 SectionPrompt objects — one per facial section.
  Each SectionPrompt contains:
    - section label
    - full prompt text (measurements + instructions + schema)
    - expected_features list (what VLM should return)

  NO images here — images are handled by FaceDescriber.
  NO VLM calls here.
  NO geometry calculations here.

INPUT:
  measurements : Dict from GeometryExtractor.compute()
  has_profile  : bool
  profile_side : "left" | "right" | None

OUTPUT:
  Dict[str, SectionPrompt]
    "section_1" → upper face prompt
    "section_2" → middle face prompt
    "section_3" → lower face prompt
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================
#  Section → Region Mapping
# =============================================================

SECTION_REGIONS = {
    "section_1": ["forehead", "eyebrows", "eyes"],
    "section_2": ["eyes",     "nose",     "cheeks"],
    "section_3": ["mouth",    "jaw",      "chin"],
}

SECTION_LABELS = {
    "section_1": "Upper Face — forehead, eyebrows, eyes",
    "section_2": "Middle Face — eyes, nose, cheeks",
    "section_3": "Lower Face — mouth, jaw, chin",
}


# =============================================================
#  Expected Features per Section
# =============================================================
#
#  These define exactly what the VLM should return.
#  Used by FaceDescriber to validate JSON output,
#  and later by feature_registry.py when introduced.

EXPECTED_FEATURES = {
    "section_1": [
        "forehead.shape",
        "forehead.lines.horizontal",
        "forehead.lines.diagonal",
        "forehead.between_eyebrow_lines",
        "eyebrows.right.shape",
        "eyebrows.right.position",
        "eyebrows.right.thickness",
        "eyebrows.left.shape",
        "eyebrows.left.position",
        "eyebrows.left.thickness",
        "eyebrows.type",
        "eyes.corner_indents",
        "eyes.puffs",
        "eyes.eyelashes",
        "eyes.bottom_lid_shape",
        "eyes.top_lid_type",
        "eyes.top_lid_cuts_pupil",
        "eyes.depth",
    ],
    "section_2": [
        "eyes.white_showing",
        "nose.tip_shape",
        "nose.nostril_size",
        "nose.nostril_shape",
        "nose.ridge",
        "nose.bump_on_bridge",
        "nose.low_septum",
        "nose.flange_marking",
        "cheeks.cheekbones",
        "cheeks.width",
        "cheeks.sunken",
        "cheeks.healer",
    ],
    "section_3": [
        "mouth.lips_fullness",
        "mouth.lip_balance",
        "mouth.cupids_bow",
        "mouth.lines",
        "mouth.teeth",
        "mouth.smile_type",
        "jaw.width",
        "jaw.pads",
        "jaw.ripples",
        "chin.front_shape",
        "chin.large",
        "chin.marks",
        "chin.cheek_marks",
    ],
}


# =============================================================
#  Region Instructions
# =============================================================

REGION_INSTRUCTIONS: Dict[str, str] = {

"forehead": """
FOREHEAD

shape — choose one:
  "rounded"  : forehead surface curves outward
  "flat"     : surface is largely flat
  "sloped"   : forehead angles backward

lines — include ONLY what is clearly visible:
  forehead_lines → horizontal: true | diagonal: true
  between_eyebrow_lines →
    single_deep_line: true | two_lines: true |
    many_lines: true | triangle_shape: true
""",

"eyebrows": """
EYEBROWS — analyze RIGHT and LEFT separately.

Per eyebrow:
  shape    → "straight" | "angled" | "curved" | "rounded"
  position → "high" | "low"
  thickness→ "thick" | "thin"

eyebrow_type → "continuous" | "separated" | "bushy"
""",

"eyes": """
EYES

corner_indents: small light highlights beside inner corners — true if visible
eye_puffs    : "moderate" (extra folds) | "intense" (hangs over eye) — omit if absent
eyelashes    : "thick" | "thin"
bottom_lid_shape: "straight" | "curved"
top_lid_type : "abundant" | "thin" | "no_lid"
top_lid_cuts_pupil: true if lid covers top half of pupil — omit if not
eye_depth    : "bulging" | "recessed" — omit if not clearly evident
""",

"nose": """
NOSE

tip_shape — choose one:
  "small_ball" | "skinny_tip" | "groove_in_tip" | "heart_shaped" | "flaccid"

nostril_size — only if clearly notable:
  "very_small" | "very_large" | "long_narrow" | "huge_flared"

nostril_shape — choose one:
  "round" | "rectangular" | "small_triangular"

nose_ridge — choose one:
  "no_ridge" | "high_ridge" | "high_wide_ridge"

bump_on_bridge: true if nose widens below bridge — omit if absent
low_septum   : true if divider hangs visibly low — omit if absent
flange_marking: "crease" | "groove" — omit if absent
""",

"cheeks": """
CHEEKS

cheekbones — choose one:
  "protruding" | "full"

cheek_width — choose one:
  "wide" | "narrow"

sunken: true if cheeks appear hollow — omit if absent
healer: true if face is widest beside the eyes — omit if absent
""",

"mouth": """
MOUTH

lips_fullness — choose one: "full" | "thin"
lip_balance  — choose one: "full_lower" | "full_upper" | "balanced"
cupids_bow   : true if distinct M-shape at center of upper lip — omit if absent

lines — include ONLY what is clearly visible:
  nasolabial_folds | compassion_lines |
  upper_lip_horizontal_line | upper_lip_vertical_lines | indents_at_corners

teeth — only if teeth visible:
  even | gap_between_front | big_front_teeth | crooked_bottom | buck_teeth

smile_type — only if smiling:
  "natural" | "crooked" | "gums_showing" | "lips_stretched" | "lips_together"
""",

"jaw": """
JAW

width — choose one: "wide" | "narrow"
jaw_pads   : true if flesh pads hang from jawline — omit if absent
ripples    : true if visible muscle tension — omit if absent
""",

"chin": """
CHIN

front_shape — choose one:
  "round" | "straight" | "pointed" | "very_pointed"

large: true if chin is strong and well-defined — omit if absent

chin_marks — include ONLY what is clearly visible:
  chin_dimple | chin_cleft | bumpy_chin | arched_line

cheek_marks — include ONLY what is clearly visible:
  cheek_dimples | diagonal_lines_left | diagonal_lines_right
""",

}


# =============================================================
#  Global Rules (injected into every prompt)
# =============================================================

GLOBAL_RULES = """
STRICT RULES:
1. NEVER use: average, moderate, normal, typical, medium
   — unless it is the only accurate option.
2. Features that are absent must be COMPLETELY OMITTED from JSON.
   Do NOT return {"value": null} or {"value": false}.
3. confidence: 0.0 = uncertain, 1.0 = clearly visible.
4. Return VALID JSON ONLY. No markdown. No explanation. No preamble.
5. Describe ONLY what is directly visible. Do NOT infer personality.
6. Treat all pre-calculated measurements as ground truth.
   Do not visually re-estimate anything already measured.
"""


# =============================================================
#  SectionPrompt
# =============================================================

@dataclass
class SectionPrompt:
    """
    Ready-to-use prompt for one facial section.

    section_id        : "section_1" | "section_2" | "section_3"
    section_label     : human-readable label
    regions           : list of regions this section covers
    prompt            : full prompt text (sent alongside image to VLM)
    expected_features : feature paths VLM should return
                        used to validate/filter response JSON
    has_profile       : whether image contains a profile half
    profile_side      : "left" | "right" | None
    """
    section_id        : str
    section_label     : str
    regions           : List[str]
    prompt            : str
    expected_features : List[str]     = field(default_factory=list)
    has_profile       : bool          = False
    profile_side      : Optional[str] = None


# =============================================================
#  Prompt Builder
# =============================================================

class PromptBuilder:
    """
    Builds 3 SectionPrompt objects.

    Takes geometry measurements + profile metadata.
    Does NOT receive or handle images.

    Usage:
        builder = PromptBuilder(
            measurements = geometry_extractor.compute(),
            has_profile  = True,
            profile_side = "left",
        )
        prompts = builder.build()
        # prompts["section_1"] → SectionPrompt
    """

    def __init__(
        self,
        measurements : Dict,
        has_profile  : bool          = False,
        profile_side : Optional[str] = None,
    ):
        self.measurements = measurements
        self.has_profile  = has_profile
        self.profile_side = profile_side

    def build(self) -> Dict[str, SectionPrompt]:
        return {
            sid: self._build_one(sid)
            for sid in ["section_1", "section_2", "section_3"]
        }

    def _build_one(self, section_id: str) -> SectionPrompt:
        regions  = SECTION_REGIONS[section_id]
        label    = SECTION_LABELS[section_id]
        features = EXPECTED_FEATURES[section_id]
        prompt   = self._assemble(section_id, regions, label)

        return SectionPrompt(
            section_id        = section_id,
            section_label     = label,
            regions           = regions,
            prompt            = prompt,
            expected_features = features,
            has_profile       = self.has_profile,
            profile_side      = self.profile_side,
        )

    def _assemble(
        self,
        section_id : str,
        regions    : List[str],
        label      : str,
    ) -> str:
        parts = []

        # ── Role ─────────────────────────────────────────────
        parts.append(
            "You are a precise facial morphology analyst.\n"
            "Describe visible facial features using exact visual observations.\n"
            "Do NOT infer personality, emotions, or character."
        )

        # ── Image layout ──────────────────────────────────────
        if self.has_profile and self.profile_side:
            parts.append(
                f"IMAGE LAYOUT:\n"
                f"  TOP    : Front view — {label}\n"
                f"  BOTTOM : {self.profile_side.capitalize()} profile view "
                f"of the same region\n"
                f"  Separator: white horizontal line between top and bottom\n\n"
                f"Use profile view for depth features (eye depth, nose bridge, chin projection).\n"
                f"Use front view for all other features."
            )
        else:
            parts.append(
                f"IMAGE LAYOUT:\n"
                f"  Front view only — {label}\n\n"
                f"No profile image available.\n"
                f"Omit depth features if not clearly determinable from front."
            )

        # ── Geometry — ground truth ───────────────────────────
        geo = self.measurements.get(section_id, {})
        if geo:
            lines = ["PRE-CALCULATED MEASUREMENTS (ground truth — do not re-estimate):"]
            for k, v in geo.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for k2, v2 in v.items():
                        lines.append(f"    {k2}: {v2}")
                else:
                    lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        # ── Region instructions ───────────────────────────────
        parts.append(f"REGIONS TO ANALYZE:\n")
        for region in regions:
            instr = REGION_INSTRUCTIONS.get(region, "")
            if instr:
                parts.append(instr.strip())

        # ── Global rules ──────────────────────────────────────
        parts.append(GLOBAL_RULES.strip())

        # ── Output schema ─────────────────────────────────────
        parts.append(
            'OUTPUT FORMAT:\n'
            'JSON object: region → feature → {"value": "...", "confidence": 0.0}\n\n'
            'Example:\n'
            '{\n'
            '  "forehead": {\n'
            '    "shape": {"value": "rounded", "confidence": 0.9}\n'
            '  },\n'
            '  "eyes": {\n'
            '    "eyelashes": {"value": "thick", "confidence": 0.85}\n'
            '  }\n'
            '}\n\n'
            'Include ONLY regions and features that are visible in this section.\n'
            'Omit absent features entirely.'
        )

        return "\n\n".join(parts)

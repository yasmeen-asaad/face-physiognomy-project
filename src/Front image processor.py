"""
=============================================================
  Face Physiognomy Project — Front Image Processor
=============================================================

RESPONSIBILITY:
  - Detect face in front image
  - Run MediaPipe FaceMesh → get 468 landmarks
  - Return face_crop + crop_landmarks

  No geometry calculations.
  No section splitting.
  No VLM calls.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DetectionStatus(Enum):
    VALID            = "valid"
    NO_FACE          = "no_face_detected"
    MULTIPLE_FACES   = "multiple_faces"
    TOO_SMALL        = "image_too_small"


@dataclass
class FrontImageResult:
    """
    Output of FrontImageProcessor.

    face_crop       : cropped face image (BGR numpy array)
    crop_landmarks  : List[(x,y)] in crop-pixel space — ready for GeometryExtractor
    face_bbox       : (x, y, w, h) in original image
    status          : DetectionStatus
    is_valid        : True if detection succeeded
    message         : human-readable description
    """
    status         : DetectionStatus
    is_valid       : bool
    message        : str
    face_crop      : Optional[np.ndarray]             = None
    crop_landmarks : Optional[List[Tuple[int, int]]]  = None
    face_bbox      : Optional[Tuple[int,int,int,int]] = None


class FrontImageProcessor:
    """
    Detects and crops the face from a frontal image.
    Returns face_crop + crop_landmarks for downstream modules.

    Usage:
        processor = FrontImageProcessor()
        result    = processor.process(image_bgr)
        if result.is_valid:
            # pass to GeometryExtractor and FaceSectionBuilder
    """

    MIN_FACE_SIZE_PX   = 80
    CROP_PADDING_RATIO = 0.30
    DETECTION_CONF     = 0.5
    PRESENCE_CONF      = 0.5

    def __init__(self):
        self._init_mediapipe()

    def _init_mediapipe(self):
        import os, urllib.request
        model_file = "face_landmarker.task"
        if not os.path.exists(model_file):
            print("Downloading face_landmarker.task ...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            urllib.request.urlretrieve(url, model_file)

        base_opts = python.BaseOptions(model_asset_path=model_file)
        opts = vision.FaceLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = vision.RunningMode.IMAGE,
            num_faces                     = 2,
            min_face_detection_confidence = self.DETECTION_CONF,
            min_face_presence_confidence  = self.PRESENCE_CONF,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        print("FrontImageProcessor ready.")

    def process(self, image_bgr: np.ndarray) -> FrontImageResult:
        full_h, full_w = image_bgr.shape[:2]

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data         = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        )
        detection = self.landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return FrontImageResult(
                status=DetectionStatus.NO_FACE, is_valid=False,
                message="No face detected."
            )
        if len(detection.face_landmarks) > 1:
            return FrontImageResult(
                status=DetectionStatus.MULTIPLE_FACES, is_valid=False,
                message=f"{len(detection.face_landmarks)} faces detected. Provide exactly one."
            )

        raw_lm = detection.face_landmarks[0]

        # Bounding box from landmark extents
        xs = [int(lm.x * full_w) for lm in raw_lm]
        ys = [int(lm.y * full_h) for lm in raw_lm]
        fx, fy   = min(xs), min(ys)
        fbw, fbh = max(xs) - fx, max(ys) - fy

        if fbw < self.MIN_FACE_SIZE_PX or fbh < self.MIN_FACE_SIZE_PX:
            return FrontImageResult(
                status=DetectionStatus.TOO_SMALL, is_valid=False,
                message=f"Face too small ({fbw}×{fbh}px)."
            )

        # Crop with asymmetric padding
        # Extra top padding to include more forehead/hairline
        pad_x  = int(fbw * self.CROP_PADDING_RATIO)
        pad_top = int(fbh * 0.40)   # generous top for hairline
        pad_bot = int(fbh * 0.15)

        cx1 = max(0, fx - pad_x)
        cy1 = max(0, fy - pad_top)
        cx2 = min(full_w, fx + fbw + pad_x)
        cy2 = min(full_h, fy + fbh + pad_bot)

        face_crop = image_bgr[cy1:cy2, cx1:cx2].copy()
        crop_h, crop_w = face_crop.shape[:2]

        # Remap landmarks to crop-pixel space
        # pixel_in_crop = pixel_in_full - crop_origin
        crop_landmarks: List[Tuple[int,int]] = []
        for lm in raw_lm:
            px = int(lm.x * full_w) - cx1
            py = int(lm.y * full_h) - cy1
            px = max(0, min(px, crop_w - 1))
            py = max(0, min(py, crop_h - 1))
            crop_landmarks.append((px, py))

        return FrontImageResult(
            status         = DetectionStatus.VALID,
            is_valid       = True,
            message        = "Face detected successfully.",
            face_crop      = face_crop,
            crop_landmarks = crop_landmarks,
            face_bbox      = (fx, fy, fbw, fbh),
        )

    def __del__(self):
        if hasattr(self, "landmarker"):
            try: self.landmarker.close()
            except: pass

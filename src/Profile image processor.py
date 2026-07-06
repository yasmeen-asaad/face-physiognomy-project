"""
=============================================================
  Face Physiognomy Project — Profile Image Processor
=============================================================

RESPONSIBILITY:
  - Detect face in profile image
  - Crop the face
  - NO landmarks (MediaPipe unreliable on profile)
  - NO geometry measurements
  - Side (left/right) comes from user input — not auto-detected

WHY no auto-detection of side?
  Selfie cameras on most phones mirror the image.
  Auto-detection would require landmark analysis that is
  unreliable on profile faces, adding complexity and error risk.
  User input is the simplest and most reliable solution.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProfileImageResult:
    """
    Output of ProfileImageProcessor.

    profile_crop : cropped face image (BGR numpy array)
    side         : "left" or "right" — from user input
    is_valid     : True if face was detected
    message      : human-readable description
    """
    is_valid     : bool
    message      : str
    profile_crop : Optional[np.ndarray] = None
    side         : Optional[str]        = None


class ProfileImageProcessor:
    """
    Detects and crops the face from a profile image.
    No landmarks — uses OpenCV face detection only.

    Usage:
        processor = ProfileImageProcessor()
        result    = processor.process(image_bgr, side="left")
        if result.is_valid:
            # pass profile_crop to FaceSectionBuilder
    """

    PADDING_RATIO  = 0.20
    MIN_FACE_SIZE  = 60
    SCALE_FACTOR   = 1.1
    MIN_NEIGHBORS  = 4

    def __init__(self):
        # OpenCV Haar Cascade — works reasonably well on profile faces
        # and requires no extra download or model file
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Profile-specific cascade — better for side faces
        self.profile_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        print("ProfileImageProcessor ready.")

    def process(
        self,
        image_bgr : np.ndarray,
        side      : Optional[str] = None,   # "left" or "right" from user
    ) -> ProfileImageResult:
        """
        Detect and crop face from profile image.

        Args:
            image_bgr : BGR numpy array
            side      : "left" or "right" — provided by user, not auto-detected
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]

        # Try profile cascade first, fall back to frontal cascade
        faces = self.profile_detector.detectMultiScale(
            gray,
            scaleFactor  = self.SCALE_FACTOR,
            minNeighbors = self.MIN_NEIGHBORS,
            minSize      = (self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
        )

        if len(faces) == 0:
            # Try flipping the image — handles mirrored profiles
            gray_flipped = cv2.flip(gray, 1)
            faces = self.profile_detector.detectMultiScale(
                gray_flipped,
                scaleFactor  = self.SCALE_FACTOR,
                minNeighbors = self.MIN_NEIGHBORS,
                minSize      = (self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
            )
            if len(faces) > 0:
                # Found on flipped image — flip the original too
                image_bgr = cv2.flip(image_bgr, 1)
                h, w = image_bgr.shape[:2]

        if len(faces) == 0:
            # Final fallback: try frontal cascade (user may upload 3/4 view)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor  = self.SCALE_FACTOR,
                minNeighbors = self.MIN_NEIGHBORS,
                minSize      = (self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
            )

        if len(faces) == 0:
            return ProfileImageResult(
                is_valid = False,
                message  = "No face detected in profile image. "
                           "Ensure the face is clearly visible from the side.",
                side     = side,
            )

        # Take the largest detected face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        pad_x = int(fw * self.PADDING_RATIO)
        pad_y = int(fh * self.PADDING_RATIO)

        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)

        profile_crop = image_bgr[y1:y2, x1:x2].copy()

        return ProfileImageResult(
            is_valid     = True,
            message      = f"Profile face detected ({fw}×{fh}px).",
            profile_crop = profile_crop,
            side         = side,
        )

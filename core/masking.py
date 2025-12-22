from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import mediapipe as mp

@dataclass
class MaskResult:
    mask: Image.Image
    overlay: Image.Image
    ok: bool
    message: str

class AutoMasker:
    """Auto-generate an upper-body clothing mask using MediaPipe pose keypoints.

    This is a **prototype-friendly** approach (no datasets), not perfect.
    Works best for front-facing, full upper body visible photos.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _pt(self, landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def make_upper_body_mask(self, person_rgb: Image.Image) -> MaskResult:
        img = np.array(person_rgb.convert("RGB"))
        h, w = img.shape[:2]

        # MediaPipe expects RGB
        results = self.pose.process(img)

        if not results.pose_landmarks:
            empty = Image.new("L", (w, h), 0)
            return MaskResult(mask=empty, overlay=person_rgb, ok=False, message="Pose not detected. Try a clearer photo.")

        lm = results.pose_landmarks.landmark
        # Landmarks we need
        LS = self._pt(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
        RS = self._pt(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
        LH = self._pt(lm[self.mp_pose.PoseLandmark.LEFT_HIP], w, h)
        RH = self._pt(lm[self.mp_pose.PoseLandmark.RIGHT_HIP], w, h)
        LE = self._pt(lm[self.mp_pose.PoseLandmark.LEFT_ELBOW], w, h)
        RE = self._pt(lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW], w, h)

        NOSE = self._pt(lm[self.mp_pose.PoseLandmark.NOSE], w, h)

        # Build a torso polygon (shoulders + hips) with a bit of margin
        def expand(a, b, factor=0.12):
            ax, ay = a; bx, by = b
            vx, vy = ax - bx, ay - by
            return int(ax + vx * factor), int(ay + vy * factor)

        LS2 = expand(LS, RS)
        RS2 = expand(RS, LS)
        LH2 = expand(LH, RH, 0.10)
        RH2 = expand(RH, LH, 0.10)

        torso = np.array([LS2, RS2, RH2, LH2], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [torso], 255)

        # Add upper arms (shoulder->elbow tubes)
        def add_arm(p1, p2, thickness):
            cv2.line(mask, p1, p2, 255, thickness=thickness, lineType=cv2.LINE_AA)

        torso_width = int(max(20, 0.20 * np.linalg.norm(np.array(LS) - np.array(RS))))
        add_arm(LS, LE, torso_width)
        add_arm(RS, RE, torso_width)

        # Exclude head / face region (simple circle around nose)
        head_r = int(max(25, 0.12 * np.linalg.norm(np.array(LS) - np.array(RS))))
        cv2.circle(mask, NOSE, head_r, 0, thickness=-1)

        # Smooth & slightly dilate for better inpainting coverage
        k = max(7, (torso_width // 2) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)

        # Overlay preview
        overlay = img.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255
        alpha = (mask.astype(np.float32) / 255.0) * 0.35
        overlay = (overlay * (1 - alpha[..., None]) + red * alpha[..., None]).astype(np.uint8)

        return MaskResult(
            mask=Image.fromarray(mask, mode="L"),
            overlay=Image.fromarray(overlay, mode="RGB"),
            ok=True,
            message="Auto-mask created.",
        )

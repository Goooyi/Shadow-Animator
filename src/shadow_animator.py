import cv2
import numpy as np

from .utils import (
    apply_noise_movement,
    apply_simplex_noise_movement,
    apply_motion_blur,
    interpolate_keyframes,
    apply_depth_deformation,
    resize_shadow_mask,
    otsu_threshold,
    adaptive_threshold,
    simple_grayscale,
    clean_up_mask
)


class ShadowAnimator:
    def __init__(
        self,
        shadow_mask,
        background_image,
        depth_map,
        output_path,
        fps=30,
        num_frames=180,
    ):
        self.shadow_mask = shadow_mask
        self.background_image = background_image
        self.depth_map = depth_map
        self.output_path = output_path
        self.fps = fps
        self.num_frames = num_frames
        self.keyframes = []

    def add_keyframe(self, frame_number, transform):
        self.keyframes.append({"frame": frame_number, "transform": transform})

    def generate_video(
        self,
        preprocess_mask=False,
        auto_threshold_method="otsu", # otsu, adaptive or simple_grayscale
        block_size=11,
        c=2,
        mask_threshold=127,
        apply_noise=True,
        apply_blur=True,
        apply_keyframes=True,
        apply_depth=True,
    ):
        # Check if the shadow mask is a video or an image
        if isinstance(self.shadow_mask, str) and (
            self.shadow_mask.endswith(".mp4") or self.shadow_mask.endswith(".avi")
        ):
            # Open the shadow mask video
            shadow_mask_cap = cv2.VideoCapture(self.shadow_mask)
            shadow_mask_fps = shadow_mask_cap.get(cv2.CAP_PROP_FPS)

            # Calculate the frame step based on the desired FPS
            frame_step = int(shadow_mask_fps / self.fps)
        else:
            # Read the shadow mask image
            shadow_mask_img = cv2.imread(self.shadow_mask, 0)

        # Create output video writer
        output_video = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.background_image.shape[1], self.background_image.shape[0]),
        )
        shadow_output_video = cv2.VideoWriter(
            self.output_path.replace(".mp4", "_shadow.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.background_image.shape[1], self.background_image.shape[0]),
        )

        frame_number = 0
        while frame_number < self.num_frames:
            if isinstance(self.shadow_mask, str) and (
                self.shadow_mask.endswith(".mp4") or self.shadow_mask.endswith(".avi")
            ):
                ret, shadow_mask = shadow_mask_cap.read()
                if not ret:
                    shadow_mask_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, shadow_mask = shadow_mask_cap.read()

                # Skip frames based on the desired FPS
                for _ in range(frame_step - 1):
                    shadow_mask_cap.read()
            else:
                shadow_mask = shadow_mask_img.copy()

            # Preprocess the shadow mask (optional)
            if preprocess_mask:
                if auto_threshold_method == "otsu":
                    shadow_mask = otsu_threshold(shadow_mask)
                elif auto_threshold_method == "adaptive":
                    shadow_mask = adaptive_threshold(
                        shadow_mask, block_size=block_size, c=c
                    )
                elif auto_threshold_method == 'grayscale':
                    shadow_mask = simple_grayscale(shadow_mask)
                else:
                    raise ValueError(f'the given mehtod {auto_threshold_method} is not supported')
            # Ensure the mask is properly inverted (shadows are white, background is black)
            shadow_mask = clean_up_mask(shadow_mask)

            # Resize the shadow mask to match the background image dimensions
            resized_shadow_mask = resize_shadow_mask(
                shadow_mask,
                (self.background_image.shape[1], self.background_image.shape[0]),
                resize_method="stretch",
            )

            # Apply noise movement
            if apply_noise:
                resized_shadow_mask = apply_noise_movement(
                    resized_shadow_mask, frame_number
                )

            # Apply motion blur
            if apply_blur:
                motion_angle = 45  # Adjust the motion angle as needed
                motion_distance = 10  # Adjust the motion distance as needed
                resized_shadow_mask = apply_motion_blur(
                    resized_shadow_mask, motion_angle, motion_distance
                )

            # Apply keyframe animation
            if apply_keyframes:
                transform = interpolate_keyframes(self.keyframes, frame_number)
                resized_shadow_mask = cv2.warpAffine(
                    resized_shadow_mask,
                    transform,
                    (resized_shadow_mask.shape[1], resized_shadow_mask.shape[0]),
                )

            # Apply depth-based deformation
            if apply_depth:
                resized_shadow_mask = apply_depth_deformation(
                    resized_shadow_mask, self.depth_map
                )

            # Create a RGB version of the processed shadow mask
            rgb_mask = cv2.cvtColor(resized_shadow_mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("/workspaces/Shadow-Animator/output/mask_clean_rgb.jpg", rgb_mask)

            inverted_rgb_mask = cv2.bitwise_not(rgb_mask)
            normalized_mask = inverted_rgb_mask.astype(float) / 255.0
            shadow_intensity = 0.4  # Adjust this value between 0 and 1 (0 = no shadow, 1 = full shadow)
            normalized_mask = 1 - (1 - normalized_mask) * shadow_intensity

            frame_with_shadow = cv2.multiply(self.background_image.astype(float), normalized_mask)
            frame_with_shadow = np.clip(frame_with_shadow, 0, 255).astype(np.uint8)
            cv2.imwrite("/workspaces/Shadow-Animator/output/mask_clean_compose.jpg", frame_with_shadow)

            # Write the processed shadow mask to the shadow output video
            shadow_output_video.write(rgb_mask)
            # # Write the frame with shadow to the output video
            output_video.write(frame_with_shadow)

            frame_number += 1

        # Release the video objects
        if isinstance(self.shadow_mask, str) and (
            self.shadow_mask.endswith(".mp4") or self.shadow_mask.endswith(".avi")
        ):
            shadow_mask_cap.release()
        output_video.release()
        shadow_output_video.release()

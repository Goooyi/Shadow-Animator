import cv2
import numpy as np

from src.shadow_animator import ShadowAnimator

# Usage example
shadow_animator = ShadowAnimator(
    shadow_mask="/workspaces/Shadow-Animator/data/shadow_mask_image/shadow_11.webp",
    background_image=cv2.imread(
        "/workspaces/Shadow-Animator/data/background_image/living_room_mockup_2.webp"
    ),
    # depth_map=cv2.imread('depth_map.png', 0),
    depth_map=None,
    output_path="/workspaces/Shadow-Animator/output/output_video_withKeyFrame.mp4",
    fps=24,
    num_frames=180,
)

# Add keyframes for animation
# TODO: refine movement
shadow_animator.add_keyframe(0, np.eye(2, 3))
shadow_animator.add_keyframe(60, np.array([[0.9, 0.1, 20], [-0.1, 0.9, 10]]))
shadow_animator.add_keyframe(120, np.array([[1.0, 0, -10], [0, 1.0, -5]]))

# Generate the output video
shadow_animator.generate_video(
    preprocess_mask=True,
    noise_gen_method="simple",
    auto_threshold_method="grayscale",
    apply_noise=True,
    apply_blur=False,
    apply_keyframes=True,
    apply_depth=False,
)

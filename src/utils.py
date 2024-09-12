import cv2
import numpy as np
from noise import snoise3


def clean_up_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def otsu_threshold(shadow_mask):
    if len(shadow_mask.shape) == 3:
        shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    _, processed_mask = cv2.threshold(
        shadow_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.bitwise_not(processed_mask)


def adaptive_threshold(shadow_mask, block_size=11, c=2):
    if len(shadow_mask.shape) == 3:
        shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    processed_mask = cv2.adaptiveThreshold(
        shadow_mask,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return cv2.bitwise_not(processed_mask)


def simple_grayscale(shadow_mask):
    # Convert to grayscale if it's not already
    if len(shadow_mask.shape) == 3:
        shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    # Optionally, adjust contrast
    shadow_mask = cv2.convertScaleAbs(shadow_mask, alpha=1.5, beta=0)
    return cv2.bitwise_not(shadow_mask)


# simple sine-cosine based noise
def apply_noise_movement(
    shadow_mask, frame_number, noise_scale=0.01, noise_speed=0.05, movement_strength=5
):
    rows, cols = shadow_mask.shape
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    # Generate noise using vectorized operations
    noise = np.sin(x * noise_scale + frame_number * noise_speed) * np.cos(
        y * noise_scale + frame_number * noise_speed
    )

    # Normalize noise to [-1, 1] range
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1

    # Create movement map
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = map_x.astype(np.float32) + noise * movement_strength
    map_y = map_y.astype(np.float32) + noise * movement_strength

    # Remap the shadow mask
    # Combine map_x and map_y into a single 2-channel array
    map_xy = np.dstack((map_x, map_y)).astype(np.float32)
    moved_shadow = cv2.remap(
        shadow_mask, map_xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    return moved_shadow.astype(np.uint8)


# TODO: better performance
def apply_simplex_noise_movement(
    shadow_mask, frame_number, noise_scale=0.01, noise_speed=0.01, movement_strength=3
):
    rows, cols = shadow_mask.shape

    # Create coordinate arrays
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    x *= noise_scale
    y *= noise_scale
    t = frame_number * noise_speed

    # Vectorize the snoise3 function
    vsnoise3 = np.vectorize(lambda x, y, z: snoise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0))

    # Generate noise using vectorized snoise3
    noise = vsnoise3(x, y, t)

    # Normalize noise to [-1, 1] range
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1

    # Create movement map
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = map_x.astype(np.float32) + noise * movement_strength
    map_y = map_y.astype(np.float32) + noise * movement_strength

    # Combine map_x and map_y into a single 2-channel array
    map_xy = np.dstack((map_x, map_y)).astype(np.float32)

    # Remap the shadow mask
    moved_shadow = cv2.remap(shadow_mask, map_xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return moved_shadow.astype(np.uint8)


def apply_motion_blur(shadow_mask, motion_angle, motion_distance):
    # Create motion blur kernel
    kernel_size = int(motion_distance) * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(
        kernel,
        cv2.getRotationMatrix2D(
            (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), motion_angle, 1.0
        ),
        (kernel_size, kernel_size),
    )
    kernel = kernel / np.sum(kernel)

    # Apply motion blur to the shadow mask
    blurred_mask = cv2.filter2D(shadow_mask, -1, kernel)
    return blurred_mask

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def interpolate_keyframes(keyframes, frame_number):
    # Find the keyframes surrounding the current frame
    previous_keyframe = None
    next_keyframe = None
    for keyframe in keyframes:
        if keyframe["frame"] <= frame_number:
            previous_keyframe = keyframe
        else:
            next_keyframe = keyframe
            break

    # Interpolate between the keyframes
    if previous_keyframe is None:
        return next_keyframe["transform"]
    elif next_keyframe is None:
        return previous_keyframe["transform"]
    else:
        t = (frame_number - previous_keyframe["frame"]) / (
            next_keyframe["frame"] - previous_keyframe["frame"]
        )
        # interpolated_transform = (
        #     previous_keyframe["transform"] * (1 - t) + next_keyframe["transform"] * t
        # )
        # return interpolated_transform
        # Apply easing function
        t_eased = ease_in_out_cubic(t)

        # Interpolate translation separately
        prev_transform = previous_keyframe["transform"]
        next_transform = next_keyframe["transform"]

        # Interpolate rotation and scaling
        interpolated_transform = prev_transform[:, :2] * (1 - t_eased) + next_transform[:, :2] * t_eased

        # Interpolate translation
        prev_translation = prev_transform[:, 2]
        next_translation = next_transform[:, 2]
        interpolated_translation = prev_translation * (1 - t_eased) + next_translation * t_eased

        # Combine interpolated components
        interpolated_transform = np.column_stack((interpolated_transform, interpolated_translation))

        return interpolated_transform


def apply_depth_deformation(shadow_mask, depth_map, depth_scale=0.01):
    # Normalize depth map to [0, 1]
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Apply depth-based deformation to the shadow mask
    rows, cols = shadow_mask.shape
    depth_mask = cv2.resize(depth_map, (cols, rows))
    deformed_mask = shadow_mask.astype(np.float32) * (1 - depth_scale * depth_mask)
    deformed_mask = np.clip(deformed_mask, 0, 255).astype(np.uint8)
    return deformed_mask


def resize_shadow_mask(shadow_mask, target_size, resize_method="stretch"):
    if resize_method == "stretch":
        resized_mask = cv2.resize(shadow_mask, target_size)
    elif resize_method == "pad":
        height, width = shadow_mask.shape
        target_height, target_width = target_size
        top = (target_height - height) // 2
        bottom = target_height - height - top
        left = (target_width - width) // 2
        right = target_width - width - left
        resized_mask = cv2.copyMakeBorder(
            shadow_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
    elif resize_method == "crop":
        height, width = shadow_mask.shape
        target_height, target_width = target_size
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        resized_mask = shadow_mask[
            top : top + target_height, left : left + target_width
        ]
    else:
        raise ValueError(f"Invalid resize method: {resize_method}")
    return resized_mask

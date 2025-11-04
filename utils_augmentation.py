# https://albumentations.ai/docs/#learning-path
# https://albumentations.ai/docs/2-core-concepts/pipelines/
# https://www.kaggle.com/code/pritishmishra/augmentation-with-albumentations bboxes
# https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/ bboxes
import albumentations as A
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib

# matplotlib.use("TkAgg")
import random
from albumentations.core.keypoints_utils import KeypointParams
from PIL import Image
import numpy as np


def augment_image_and_keypoints(img, keypoints):
    size_crop = img.shape[0]
    pipeline = create_augmentation_pipeline(num_transforms=3, size_crop=size_crop)

    # assert isinstance(img, Image.Image), f"Expected PIL.Image.Image, got {type(img)}"
    assert isinstance(keypoints, np.ndarray), f"Expected numpy.ndarray, got {type(keypoints)}"
    assert keypoints.shape[1] == 2, f"Expected keypoints with shape (N, 2), got {keypoints.shape}"

    # img = np.array(img)  # PIL → ndarray
    assert img.ndim == 3 and img.shape[2] == 3, f"Expected RGB image with 3 channels, got shape {img.shape}"

    keypoint_params = KeypointParams(format="xy")
    augmented = pipeline(image=img, keypoints=keypoints)
    img_aug = augmented["image"]
    keypoints_aug = augmented["keypoints"]
    # img_aug = Image.fromarray(img_aug)  # back to PIL if needed
    return img_aug, keypoints_aug


def get_weighted_random_pipeline():
    # Optional transforms grouped by type
    transform_pool = [
        # Geometric transforms
        (A.HorizontalFlip(p=0.5), 0.8),
        (A.VerticalFlip(p=0.5), 0.3),
        (A.Rotate(limit=(-90, 90), p=0.3), 0.2),
        # Color transforms
        (A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6), 0.5),
        (A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4), 0.5),
        # Blur / Noise / Distortion (only one applied at a time)
        (
            A.OneOf(
                [
                    A.Blur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(std_range=(0.1, 0.25), mean_range=(0, 0), per_channel=True, noise_scale_factor=1, p=1.0),
                ],
                p=0.5,
            ),
            0.5,
        ),
        # other distortions
        (A.RandomGamma(gamma_limit=(90, 110), p=0.5), 0.2),
        (A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3), 0.3),
        (A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3), 0.2),
        (A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.2), 0.1),
        (A.Perspective(scale=(0.05, 0.1), p=0.2), 0.3),
        (A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2), 0.2),
    ]

    return transform_pool


def create_augmentation_pipeline(num_transforms: int = 3, size_crop=512):
    transform_pool = get_weighted_random_pipeline()
    selected_transforms = []
    weights = [weight for _, weight in transform_pool]

    # Sample without replacement
    sampled_indices = random.choices(range(len(transform_pool)), weights=weights, k=num_transforms)

    for idx in sampled_indices:
        transform, _ = transform_pool[idx]
        selected_transforms.append(transform)

    mandatory = [
        A.RandomResizedCrop(
            size=[size_crop, size_crop], scale=[0.01, 1], ratio=[1, 1], interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, p=1
        )
    ]

    final_pipeline = mandatory + selected_transforms

    return A.Compose(
        final_pipeline,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def create_augmentation_pipeline_with_proba(transform_pool):
    selected_transforms = []
    for transform, prob in transform_pool:
        if random.random() < prob:
            selected_transforms.append(transform)

    return A.Compose(selected_transforms)


def save_original_image(save_path, image):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    plt.title("Original Image")
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    print(f"Saved figure to: {save_path}")


def save_transformed_image(save_path, transformed_image, i, transforms_text):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(transformed_image)
    plt.title("Transformed Image")
    plt.figtext(0.5, 0.01, transforms_text, wrap=True, ha="center", fontsize=12)
    plt.savefig(f"{save_path[:-4]}_{i}.jpg", bbox_inches="tight", dpi=350)
    print(f"Saved figure to: {save_path[:-4]}_{i}.jpg")
    plt.close(fig)

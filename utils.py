from PIL import Image
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch as th
import cv2
from scipy import ndimage


def pad_image(image: Image.Image) -> tuple:
    """
    Pad image with white pixels to make it square.

    Args:
        image: PIL image

    Returns:
        padded_image: PIL image
        image_mask: PIL image
    """

    pad_length = 75
    height, width = np.asarray(image).shape[:2]
    new_width = width + 2 * pad_length
    new_height = height + 2 * pad_length

    border_padding = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    inner_image_mask = Image.new(image.mode, (width, height), (0, 0, 0))

    padded_image = border_padding.copy()
    padded_image.paste(image, (pad_length, pad_length))

    image_mask = border_padding.copy()
    image_mask.paste(inner_image_mask, (pad_length, pad_length))

    return padded_image, image_mask


def get_image_file_paths() -> list:
    """
    Returns path to all images in images folder.

    Returns:
        image_file_names: List of strings.
    """

    image_file_names = glob.glob("images/*[.png|.jpg|.jpeg]")
    return image_file_names


def display_image(image: Image.Image, frame_name="Display Inage") -> None:
    """
    Display image using matplotlib.

    Args:
        image: PIL image

    Returns:
        None
    """
    plt.imshow(image)
    plt.title(frame_name)
    plt.show()


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to range [0, 1].

    Args:
        arr: Numpy array

    Returns:
        arr: Numpy array
    """

    return (arr - arr.min()) / (arr.max() - arr.min())


def normalize_alpha_matte(M: np.ndarray) -> np.ndarray:
    """
    Normalize alpha matte to range [0, 1].

    Args:
        M: Alpha matte.

    Returns:
        M_prime: Normalized alpha matte.
    """

    pooling = th.nn.MaxPool1d(kernel_size=5, stride=1)
    M_prime = pooling(th.from_numpy(M).float())
    M_prime = cv2.resize(
        np.array(M_prime), M.shape[:2][::-1], interpolation=cv2.INTER_AREA
    )
    return M_prime


def depth_based_soft_foreground_pixel_visibility_map(depth: np.ndarray) -> np.ndarray:
    """
    Uses sobel gradients to generate soft foreground pixel visibility map.

    Args:
        depth: Depth map.

    Returns:
        pixel_visibility_map: Soft foreground pixel visibility map.
    """

    sobel_gradient_x = ndimage.sobel(depth, axis=0, mode="constant")
    sobel_gradient_y = ndimage.sobel(depth, axis=1, mode="constant")
    sobel_gradient = np.hypot(sobel_gradient_x, sobel_gradient_y)
    beta = 3
    pixel_visibility_map = np.exp(-beta * np.square(sobel_gradient))
    return pixel_visibility_map


if __name__ == "__main__":
    image_names = get_image_file_paths()
    for image_name in image_names:
        image = Image.open(image_name)
        padded_image, image_mask = pad_image(image)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(padded_image)
        axarr[1].imshow(image_mask)
        plt.show()

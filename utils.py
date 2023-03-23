from PIL import Image
import numpy as np
import glob
from matplotlib import pyplot as plt


def pad_image(image: Image.Image) -> tuple:
    """
    Pad image with white pixels to make it square.

    Args:
        image: PIL image
    Returns:
        padded_image: PIL image
        image_mask: PIL image
    """

    pad_length = 100
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


def get_image_file_names() -> list:
    """
    Returns path to all images in images folder.

    Returns:
        image_file_names: List of strings
    """

    image_file_names = glob.glob("images/*[.png|.jpg|.jpeg]")
    return image_file_names


def display_image(image):
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    image_names = get_image_file_names()
    for image_name in image_names:
        image = Image.open(image_name)
        padded_image, image_mask = pad_image(image)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(padded_image)
        axarr[1].imshow(image_mask)
        plt.show()

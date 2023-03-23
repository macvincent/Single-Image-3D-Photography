from utils import (
    get_image_file_paths,
    pad_image,
    normalize_alpha_matte,
    normalize_array,
    depth_based_soft_foreground_pixel_visibility_map,
)
from mesh import create_mesh
import os
from PIL import Image
from models import image_completion_model, matting_model, monocular_depth_model
import numpy as np


def main():
    file_paths = get_image_file_paths()
    for file_path in file_paths:
        image_name = os.path.split(file_path)[1].split(".")[0]
        image_path = os.path.join(os.getcwd(), file_path)
        input_image = Image.open(image_path).resize((400, 400))

        # outpaint input image to increase the field of view we can sample from.
        padded_image, image_mask = pad_image(input_image)
        outpainted_image = image_completion_model.outpaint(
            input_image, padded_image, image_mask
        )
        outpainted_depth = monocular_depth_model.get_depth_map(outpainted_image)

        # generate background image and depth from outpainted image.
        alpha_matte = matting_model.get_alpha_matte(outpainted_image)
        background_image = image_completion_model.inpaint(outpainted_image, alpha_matte)
        background_depth = monocular_depth_model.get_depth_map(background_image)

        # generate soft foregound visibility mask to predict disocclusion when sampling for novel views.
        depth_based_visibility_map = depth_based_soft_foreground_pixel_visibility_map(
            outpainted_depth
        )
        normalized_alpha_matte = normalize_alpha_matte(alpha_matte)
        soft_foregound_visibility_mask = (
            alpha_matte - normalized_alpha_matte
        ) + depth_based_visibility_map
        soft_foregound_visibility_mask = normalize_array(soft_foregound_visibility_mask)
        stacked_soft_foregound_visibility_mask = np.stack(
            (
                soft_foregound_visibility_mask,
                soft_foregound_visibility_mask,
                soft_foregound_visibility_mask,
            )
        ).transpose([1, 2, 0])
        stacked_soft_foregound_visibility_image = Image.fromarray(
            np.clip(stacked_soft_foregound_visibility_mask * 255, 0, 255).astype(
                np.uint8
            )
        )

        # generate meshes
        create_mesh(outpainted_image, outpainted_depth, image_name + "_foreground_")
        create_mesh(background_image, background_depth, image_name + "_background_")
        create_mesh(
            stacked_soft_foregound_visibility_image,
            outpainted_depth,
            image_name + "_visibility_",
        )


if __name__ == "__main__":
    main()

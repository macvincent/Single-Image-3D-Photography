import open3d as o3d
import numpy as np
from utils import normalize_array
from PIL import Image
import os
import logging
import copy
from tqdm import tqdm
import imageio
import cv2

logging.basicConfig(level=logging.INFO)


def get_3D_image_points(image: Image.Image, depth: np.ndarray) -> np.ndarray:
    """
    Get (u, v, z) coordinates of all pixels in image.

    Args:
        image: RGB image
        depth: Corresponding depth map for image

    Returns:
        points: Numpy array of shape (image_height*image_width, 3)
    """

    image_height, image_width = np.array(image).shape[:2]

    points = np.zeros((image_height * image_width, 3))

    x = np.arange(0, image_width, 1)
    y = np.arange(0, image_height, 1)

    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv, 2)
    yv = np.expand_dims(yv, 2)
    grid = np.concatenate((xv, yv), axis=2)
    grid = grid.reshape((image_height * image_width, 2))

    points[:, 0] = grid[:, 0]
    points[:, 1] = grid[:, 1]
    points[:, 2] = depth[grid[:, 0], grid[:, 1]]

    return points


def create_mesh(
    rgb_image: Image.Image, depth: np.ndarray, mesh_file_name: str
) -> o3d.geometry.TriangleMesh:
    """
    Create and save mesh from an image.

    Args:
        image: RGB image
        depth: Corresponding depth map for image
        mesh_file_name: Name of mesh file

    Returns:
        mesh: Open3D mesh
    """

    # Create meshes directory if it doesn't exist.
    os.makedirs("meshes", exist_ok=True)

    # Generate grid.
    logging.info("Generate grid.")
    points = get_3D_image_points(rgb_image, depth)
    points_uv = points[:, :2].astype(np.int32).copy()
    colors = np.asarray(rgb_image)[points_uv[:, 0], points_uv[:, 1]]

    # Normalize points and colors.
    points[:, 0] = normalize_array(points[:, 0])
    points[:, 1] = normalize_array(points[:, 1])
    points[:, 2] = normalize_array(points[:, 2])
    colors = normalize_array(colors)

    # Creating point cloud.
    logging.info("Creating point cloud.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    rotate_to_th_axis = pcd.get_rotation_matrix_from_xyz((0, np.pi, -np.pi / 2))
    pcd.rotate(rotate_to_th_axis, center=(0, 0, 0))

    # Estimating normals.
    logging.info("Estimating normals.")
    pcd.estimate_normals()

    # Orienting normals. Removing this has a negative effect on the quality of our ouput mesh.
    logging.info("Orienting normals.")
    pcd.orient_normals_consistent_tangent_plane(15)

    # Creating Mesh.
    logging.info("Creating normals.")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    # Save Mesh.
    logging.info("Saving mesh.")
    mesh_path = os.path.join(os.getcwd(), "meshes", f"{mesh_file_name}mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)


def get_rotated_frame(mesh, rotate_angle):
    # TODO: Sample different trajectory types
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz(
        (0, rotate_angle, 0)
    )
    rotated_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(rotated_mesh)
    vis.update_geometry(rotated_mesh)
    vis.poll_events()
    vis.update_renderer()

    # Run the visualizer
    color_image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return np.asarray(color_image)


def sample_novel_views(image_name, config):
    # Create meshes directory if it doesn't exist.
    os.makedirs("meshes", exist_ok=True)
    logging.info("Sampling novel views.")

    foreground_visibility_mesh_path = os.path.join(
        os.getcwd(), "meshes", f"{image_name}_visibility_mesh.ply"
    )
    foreground_visibility_mesh = o3d.io.read_triangle_mesh(
        foreground_visibility_mesh_path
    )

    background_mesh_path = os.path.join(
        os.getcwd(), "meshes", f"{image_name}_background_mesh.ply"
    )
    background_mesh = o3d.io.read_triangle_mesh(background_mesh_path)

    foreground_mesh_path = os.path.join(
        os.getcwd(), "meshes", f"{image_name}_foreground_mesh.ply"
    )
    foreground_mesh = o3d.io.read_triangle_mesh(foreground_mesh_path)

    num_frames_in_output_video = config.num_frames_in_output_video
    angle_range = [-10, 10]
    angle_delta = (angle_range[1] - angle_range[0]) / num_frames_in_output_video
    output_frames = []
    for i in tqdm(range(num_frames_in_output_video)):
        rotate_angle = (angle_range[0] + angle_delta * i) * np.pi / 180
        background_image = get_rotated_frame(background_mesh, rotate_angle) * 255
        foreground_image = get_rotated_frame(foreground_mesh, rotate_angle) * 255
        foreground_visibility_map = get_rotated_frame(
            foreground_visibility_mesh, rotate_angle
        )

        novel_view = (
            foreground_visibility_map * foreground_image
            + (1 - foreground_visibility_map) * background_image
        )
        novel_view = cv2.flip(
            np.clip(np.asarray(novel_view), 0, 255).astype(np.uint8), 1
        )

        output_frames.append(novel_view)

    for i in tqdm(range(num_frames_in_output_video - 1, -1, -1)):
        rotate_angle = (angle_range[0] + angle_delta * i) * np.pi / 180
        background_image = get_rotated_frame(background_mesh, rotate_angle) * 255
        foreground_image = get_rotated_frame(foreground_mesh, rotate_angle) * 255
        foreground_visibility_map = get_rotated_frame(
            foreground_visibility_mesh, rotate_angle
        )

        novel_view = (
            foreground_visibility_map * foreground_image
            + (1 - foreground_visibility_map) * background_image
        )
        novel_view = cv2.flip(
            np.clip(np.asarray(novel_view), 0, 255).astype(np.uint8), 1
        )

        output_frames.append(novel_view)

    os.makedirs("outputs", exist_ok=True)
    if config.save_output_in_gif_format:
        output_path = os.path.join(os.getcwd(), "outputs", f"{image_name}.gif")
        imageio.mimsave(output_path, output_frames)

    if config.save_output_in_mp4_format:
        output_path = os.path.join(os.getcwd(), "outputs", f"{image_name}.mp4")
        imageio.mimsave(output_path, output_frames)

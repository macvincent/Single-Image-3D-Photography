import open3d as o3d
import numpy as np
from utils import normalize_array
from PIL import Image
import os
import logging

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

    # Creating Point Cloud.
    logging.info("Creating Point Cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    rotate_to_th_axis = pcd.get_rotation_matrix_from_xyz((0, np.pi, -np.pi / 2))
    pcd.rotate(rotate_to_th_axis, center=(0, 0, 0))

    # Estimating Normals.
    logging.info("Estimating Normals...")
    pcd.estimate_normals()

    # Orienting Normals.
    logging.info("Orienting Normals...")
    pcd.orient_normals_consistent_tangent_plane(15)

    # Creating Mesh.
    logging.info("Creating Normals...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    # Save Mesh.
    logging.info("Saving Mesh...")
    o3d.io.write_triangle_mesh(f"meshes/{mesh_file_name}mesh.ply", mesh)

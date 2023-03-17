import torch
import matplotlib.pyplot as plt
import copy
import open3d as o3d
import numpy as np
import imageio
from tqdm import tqdm

import trimesh
from PIL import Image
import io

device=torch.device("cuda:0")

def rotate_mesh(mesh, rotate_angle=-np.pi):
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((0, 0, rotate_angle))
    rotated_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([rotated_mesh])

    return rotated_mesh

def get_rotated_frame(mesh, rotate_angle):
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((rotate_angle, rotate_angle, rotate_angle))
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


# def get_trimesh(open_3d_mesh):
#     vert_features = np.array(open_3d_mesh.vertex_colors)
#     verts = np.array(open_3d_mesh.vertices)
#     faces = np.array(open_3d_mesh.triangles)
#     normals = np.array(open_3d_mesh.vertex_normals)

#     mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=vert_features)
#     mesh = trimesh.scene.Scene(mesh)
#     return mesh

# def get_rotated_frame(scene, rotate_angle):
#     # Define camera parameters
#     width = 512
#     height = 512

#     scene.set_camera(distance=0.8, angles=(rotate_angle, rotate_angle, rotate_angle), resolution=(512, 512))

#     # Render scene and save image
#     image = scene.save_image(resolution=(width, height), visible=True)
#     image = Image.open(io.BytesIO(image))
#     return np.asarray(image)[:, :, :3]

foreground_visibility_mesh = o3d.io.read_triangle_mesh('foreground_visibility_mesh.ply')
foreground_visibility_mesh = rotate_mesh(foreground_visibility_mesh)

background_mesh = o3d.io.read_triangle_mesh('background_mesh.ply')
background_mesh = rotate_mesh(background_mesh)

foreground_mesh = o3d.io.read_triangle_mesh('foreground_mesh.ply')
foreground_mesh = rotate_mesh(foreground_mesh)

method_factor = 255

# foreground_visibility_mesh = get_trimesh(foreground_visibility_mesh)
# background_mesh = get_trimesh(background_mesh)
# foreground_mesh = get_trimesh(foreground_mesh)

initial_angle = -10
frames = []
for i in tqdm(range(20)):
    rotate_angle = (initial_angle + i) * np.pi / 180
    background_image = get_rotated_frame(background_mesh, rotate_angle ) * method_factor
    foreground_image = get_rotated_frame(foreground_mesh, rotate_angle) * method_factor
    foreground_visibility_map = get_rotated_frame(foreground_visibility_mesh, rotate_angle)

    novel_view = foreground_visibility_map*foreground_image + (1-foreground_visibility_map)*background_image
    novel_view = np.clip(np.asarray(novel_view), 0, 255).astype(np.uint8)

    frames.append(novel_view)
imageio.mimsave('./movie.gif', frames)
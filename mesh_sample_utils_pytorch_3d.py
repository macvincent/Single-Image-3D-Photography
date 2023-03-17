import torch
import matplotlib.pyplot as plt
import copy
import open3d as o3d
import numpy as np
import imageio
from tqdm import tqdm
import pytorch3d
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)

device=torch.device("cuda:0")
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1,
)

R, T = look_at_view_transform(3, 0, 0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

rasterizer=MeshRasterizer(
    cameras=cameras, 
    raster_settings=raster_settings
)

def pytorch_3d_mesh(mesh, rotate_angle, get_rotated_mesh=False):
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((0, 0, rotate_angle))
    rotated_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))


    vert_features = np.array(rotated_mesh.vertex_colors)
    verts = np.array(rotated_mesh.vertices)
    faces = np.array(rotated_mesh.triangles)
    torch_mesh = None

    if get_rotated_mesh:
        torch_mesh = pytorch3d.structures.Meshes([torch.from_numpy(verts.astype(np.float32))], [torch.from_numpy(faces)])
        torch_mesh = torch_mesh.to(device)

    return torch_mesh, vert_features, faces


foreground_visibility_mesh = o3d.io.read_triangle_mesh('foreground_visibility_mesh.ply')
rotate_to_torch_axis = foreground_visibility_mesh.get_rotation_matrix_from_xyz((0, 0, -np.pi))
foreground_visibility_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))

background_mesh = o3d.io.read_triangle_mesh('background_mesh.ply')
background_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))

foreground_mesh = o3d.io.read_triangle_mesh('foreground_mesh.ply')
foreground_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))

initial_angle = -10
frames = []
for i in tqdm(range(20)):
    rotate_angle = (initial_angle + i) * np.pi / 180

    _, visibility_color, _ = pytorch_3d_mesh(foreground_visibility_mesh, rotate_angle)
    _, background_color, _ = pytorch_3d_mesh(background_mesh, rotate_angle)
    rotated_mesh, foreground_color, faces = pytorch_3d_mesh(foreground_mesh, rotate_angle, True)

    fragments = rasterizer(rotated_mesh)
    vertex_ids = faces[fragments.pix_to_face.cpu()]

    foreground_image = foreground_color[vertex_ids].mean(axis=(-2, -3))[0]*255
    background_image = background_color[vertex_ids].mean(axis=(-2, -3))[0]*255
    alpha = visibility_color[vertex_ids].mean(axis=(-2, -3))[0]

    novel_view = alpha*foreground_image + (1-alpha)*background_image
    novel_view = np.clip(np.asarray(novel_view), 0, 255).astype(np.uint8)

    frames.append(novel_view)

imageio.mimsave('./movie_pytorch.gif', frames)

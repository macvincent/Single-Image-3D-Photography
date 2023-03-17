import copy
import open3d as o3d
import numpy as np
import imageio
from tqdm import tqdm
from vispy import app, gloo, visuals, scene
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
import matplotlib.pyplot as plt

class Canvas_view():
    def __init__(self,
                 fov,
                 verts,
                 faces,
                 colors,
                 canvas_size,
                 factor=1,
                 bgcolor='gray',
                 proj='perspective',
                 ):
        self.canvas = scene.SceneCanvas(bgcolor=bgcolor, size=(canvas_size*factor, canvas_size*factor))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = proj
        self.view.camera.fov = fov
        self.mesh = visuals.Mesh(shading=None)
        self.mesh.attach(Alpha(1.0))
        self.view.add(self.mesh)
        self.tr = self.view.camera.transform
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])
        self.translate([-0.5, -0.5, 1.5])
        self.rotate(axis=[1,0,0], angle=180)
        self.view_changed()

    def translate(self, trans=[0,0,0]):
        self.tr.translate(trans)

    def rotate(self, axis=[1,0,0], angle=0):
        self.tr.rotate(axis=axis, angle=angle)

    def view_changed(self):
        self.view.camera.view_changed()

    def render(self):
        return self.canvas.render()

    def reinit_mesh(self, verts, faces, colors):
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])

    def reinit_camera(self, fov):
        self.view.camera.fov = fov
        self.view.camera.view_changed()


def rotate_mesh(mesh, rotate_angle=np.pi):
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((0, 0, rotate_angle))
    rotated_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))

    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    rotated_mesh.rotate(rotate_to_torch_axis, center=(0, 0, 0))

    return rotated_mesh


def get_vispy_canvas(open_3d_mesh):
    colors = np.array(open_3d_mesh.vertex_colors)
    verts = np.array(open_3d_mesh.vertices)
    faces = np.array(open_3d_mesh.triangles)
    normals = np.array(open_3d_mesh.vertex_normals)

    normal_canvas = Canvas_view(52,
                                verts,
                                faces,
                                colors,
                                canvas_size=512,
                                factor=2,
                                bgcolor='black',
                                proj='perspective')

    return normal_canvas

def get_rotated_frame(mesh, rotate_angle):
    rotated_mesh = copy.deepcopy(mesh)
    rotate_to_torch_axis = rotated_mesh.get_rotation_matrix_from_xyz((rotate_angle, rotate_angle, rotate_angle))
    rotated_mesh.rotate(rotate_to_torch_axis, center=(-0.5,-0.5,0))

    canvas = get_vispy_canvas(rotated_mesh)

    return canvas.render()[:, :, :3]

foreground_visibility_mesh = o3d.io.read_triangle_mesh('foreground_visibility_mesh.ply')
foreground_visibility_mesh = rotate_mesh(foreground_visibility_mesh)

background_mesh = o3d.io.read_triangle_mesh('background_mesh.ply')
background_mesh = rotate_mesh(background_mesh)

foreground_mesh = o3d.io.read_triangle_mesh('foreground_mesh.ply')
foreground_mesh = rotate_mesh(foreground_mesh)

initial_angle = -10
frames = []
for i in tqdm(range(20)):
    rotate_angle = (initial_angle + i) * np.pi / 180
    background_image = get_rotated_frame(background_mesh, rotate_angle) 
    foreground_image = get_rotated_frame(foreground_mesh, rotate_angle)
    foreground_visibility_map = get_rotated_frame(foreground_visibility_mesh, rotate_angle)
    alpha = foreground_visibility_map / foreground_visibility_map.max()
    novel_view = alpha*foreground_image + (1-alpha)*background_image
    novel_view = np.clip(np.asarray(novel_view), 0, 255).astype(np.uint8)

    frames.append(novel_view)
imageio.mimsave('./movie_vispy.gif', frames)
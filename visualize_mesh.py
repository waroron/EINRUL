import numpy as np
import argparse
import open3d as o3d
import matplotlib.pyplot as plt

def load_and_visualize_ply(ply_file_path):
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    if not mesh.has_vertex_colors():
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((np.asarray(mesh.vertices).shape[0], 3)))
    
    vertices = np.asarray(mesh.vertices)
    z_values = vertices[:, 2]
    
    z_min, z_max = z_values.min(), z_values.max()
    
    colors = plt.get_cmap('coolwarm')((z_values - z_min) / (z_max - z_min))[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    
    args = parser.parse_args()
    load_and_visualize_ply(args.input)
    
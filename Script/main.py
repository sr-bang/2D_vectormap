from pathlib import Path
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

from fit_plane import *
from error_funcs import ransac_error, msac_error, mlesac_error
from plot_results import *
from Script.transform import *
from Misc import *

if __name__ == '__main__':

    Parser = argparse.ArgumentParser()

    Parser.add_argument('--confidence', type=float, default=0.85)
    Parser.add_argument('--inlier_threshold', type=float, default=0.02)
    Parser.add_argument('--min_sample_distance', type=float, default=8.5)
    Parser.add_argument('--error_function_idx',type=int, default=0)
    Parser.add_argument('--voxel_size',type=float, default=0.005)
    Parser.add_argument('--nb_neighbors',type=int, default=35)
    Parser.add_argument('--std_ratio',type=float, default=1.0) 
    Parser.add_argument('--pointcloud_idx',type=int, default=1)


    Args = Parser.parse_args()
    confidence = Args.confidence
    inlier_threshold = Args.inlier_threshold
    min_sample_distance = Args.min_sample_distance
    error_function_idx = Args.error_function_idx
    voxel_size = Args.voxel_size
    nb_neighbors = Args.nb_neighbors
    std_ratio = Args.std_ratio
    pointcloud_idx = Args.pointcloud_idx

    error_functions = [ransac_error, msac_error, mlesac_error]

    directory_path = "../Output"
    os.makedirs(directory_path, exist_ok=True)
         
    input_path = "input.pcd"
    input_file = o3d.io.read_point_cloud(input_path)

    # Down-sample the loaded point cloud to reduce computation time
    pcd_sampled = input_file.voxel_down_sample(voxel_size)
    
    # Apply plane-fitting algorithmpcd_sampled
    best_plane, best_inliers, num_iterations = fit_plane(pcd=pcd_sampled,
                                                         confidence=confidence,
                                                         inlier_threshold=inlier_threshold,
                                                         min_sample_distance=min_sample_distance,
                                                         error_func=error_functions[error_function_idx])

    # Plot the result
    in_points = plot_dominant_plane(pcd_sampled, best_inliers, best_plane)

    # Remove statistical outlier
    pcd = remove_statistical_outlier(in_points, nb_neighbors, std_ratio)
    
    # Color Thresholding
    filtered_points, filtered_colors = color_thresholding(pcd)

    # Transforms used 
    transforms = "voxel_grid_downsampling"
    filtered_points, filtered_colors = select_transforms(filtered_points, filtered_colors, transforms)
    
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)  

    # # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualization
    vis.add_geometry(point_cloud)

    # Customize the visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2  # Adjust the point size as needed

    image = os.path.join(directory_path, 'image.png')
    vis.run()
    vis.capture_screen_image(image)
    vis.destroy_window()



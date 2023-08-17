from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]
    
    # Creating a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, coordinate_frame]
    after_ransac_points = geometries[0]

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for p in geometries:
    #     vis.add_geometry(p)
    # vc = vis.get_view_control()
    # vc.set_front([-0.3, 0.32, -0.9])
    # vc.set_lookat([-0.13, -0.15, 0.92])
    # vc.set_up([0.22, -0.89, -0.39])
    # vc.set_zoom(0.5)

    # # Customize the visualization
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size = 2  # Adjust the point size as needed
    
    # vis.run()
    # vis.destroy_window()
    return after_ransac_points

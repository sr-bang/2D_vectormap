import numpy as np
import open3d as o3d

def barycentric_interpolation(points, colors):
    tri = o3d.geometry.TriangleMesh()
    tri.vertices = o3d.utility.Vector3dVector(points)
    tri.compute_vertex_normals()
    tri.paint_uniform_color([0.5, 0.5, 0.5])
    new_points = np.asarray(tri.vertices)
    new_colors = np.asarray(tri.vertex_colors)
    return new_points, new_colors

def kernel_density_estimation(points, colors, bandwidth=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    kde = o3d.geometry.KDTreeFlann(pcd)
    densified_points = []
    densified_colors = []
    for i, point in enumerate(points):
        _, indices, _ = kde.search_radius_vector_3d(point, bandwidth)
        if len(indices) > 0:
            densified_points.append(np.mean(np.asarray(pcd.points)[indices], axis=0))
            densified_colors.append(np.mean(np.asarray(pcd.colors)[indices], axis=0))
    return np.array(densified_points), np.array(densified_colors)

def nearest_neighbor_interpolation(points, colors, k=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = []
    new_colors = []
    for i, point in enumerate(points):
        _, indices, _ = pcd_tree.search_knn_vector_3d(point, k)
        if len(indices) > 0:
            new_points.append(np.mean(np.asarray(pcd.points)[indices], axis=0))
            new_colors.append(np.mean(np.asarray(pcd.colors)[indices], axis=0))

    print("Number of points before interpolation: ", len(points))
    print("Number of points after interpolation: ", len(new_points))
    return np.array(new_points), np.array(new_colors)

def random_sampling(points, colors, num_samples=1000*100):
    num_points = len(points)
    sample_indices = np.random.choice(num_points, num_samples, replace=False)
    return points[sample_indices], colors[sample_indices]

def voxel_grid_downsampling(points, colors, voxel_size=0.07):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_downsampled.points), np.asarray(pcd_downsampled.colors)

def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return np.asarray(pcd.select_down_sample(ind).points), np.asarray(pcd.select_down_sample(ind).colors)

def radius_outlier_removal(points, colors, nb_points=20, radius=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    cl, ind = pcd.remove_radius_outlier(nb_points, radius)
    return np.asarray(pcd.select_down_sample(ind).points), np.asarray(pcd.select_down_sample(ind).colors)

def uniform_sampling(points, colors, num_samples=1000*100):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_downsampled = pcd.uniform_down_sample(num_samples)
    return np.asarray(pcd_downsampled.points), np.asarray(pcd_downsampled.colors)


def select_transforms(points, colors, transform):
    if transform == "barycentric_interpolation": #filler
        points, colors = barycentric_interpolation(points, colors)
    elif transform == "kernel_density_estimation": #smoothens alot
        points, colors = kernel_density_estimation(points, colors)
    elif transform == "nearest_neighbor_interpolation":
        points, colors = nearest_neighbor_interpolation(points, colors)
    elif transform == "random_sampling": #sparse
        points, colors = random_sampling(points, colors)
    elif transform == "voxel_grid_downsampling": #grid considerable
        points, colors = voxel_grid_downsampling(points, colors)
    elif transform == "statistical_outlier_removal":
        points, colors = statistical_outlier_removal(points, colors)
    elif transform == "radius_outlier_removal":
        points, colors = radius_outlier_removal(points, colors)
    elif transform == "uniform_sampling":
        points, colors = uniform_sampling(points, colors)
    else:
        raise ValueError("Unknown transform: %s" % transform)
    return points, colors

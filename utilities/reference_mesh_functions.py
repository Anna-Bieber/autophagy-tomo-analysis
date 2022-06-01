#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for creating rim reference meshes.

For an example of how these functions are used, check the notebook Make_rim_reference_mesh.ipynb.
Created on Thu Oct  7 09:23:13 2021
@author: anbieber
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import scipy.interpolate
from functools import partial

import pyvista as pv
import time

#from collections import namedtuple

from utility_functions import normalize_vector

from mesh_functions import mesh_find_borders, mesh_find_edges, make_2d_grid_faces

from fitting_functions import polynomial_surface_fit_v1, fit_plane


#%% Functions for constructing a toroid


def generate_spline_vectors(points):
    """Generate vectors along a spline from its ordered points."""
    vec0 = normalize_vector( points[2:] - points[:-2] )
    vec_res = np.concatenate([ np.atleast_2d( normalize_vector(points[1]-points[0]) ),
                               vec0, 
                               np.atleast_2d( normalize_vector(points[-1]-points[-2]) )], axis=0)
    return vec_res

def structured_cylinder_to_polydata(cyl, theta_res, z_res=None, all_triangles=True):
    """Turn a cylinder constructed with pyvista.CylinderStructured into PolyData."""
    if z_res == None:
        z_res = int(cyl.points.shape[0] / theta_res)
    # Construct rectangular faces
    ids_up_left = np.arange(theta_res*(z_res-1)).astype(int)
    tmp = ids_up_left.reshape((z_res-1), theta_res)
    ids_up_right = np.c_[tmp[:,1:], tmp[:,0]].flatten()
    ids_down_left = (ids_up_left + theta_res).astype(int)
    ids_down_right = (ids_up_right + theta_res).astype(int)
    rectangle_ids = np.c_[ids_up_left, ids_up_right, ids_down_left, ids_down_right]
    if not all_triangles:
        faces = np.c_[4*np.ones_like(ids_up_left), rectangle_ids]
    else :
        ids_tri_0 = [0,1,2]
        ids_tri_1 = [1,2,3]
        # put together triangle ids, always alternating between upper and lower triangle
        triangle_ids = np.c_[ rectangle_ids[:,ids_tri_0], rectangle_ids[:,ids_tri_1] ].reshape(-1,3)
        faces = np.c_[3*np.ones(triangle_ids.shape[0]), triangle_ids].astype(int)
    # Make mesh
    mesh = pv.PolyData(cyl.points, faces)
    return mesh
        
        

def make_toroid_around_spline(spline_points, radius, theta_res=72, return_intermediates=False):
    """Given the ordered points of a spline, construct a toroid around the spline."""        
    # Step 1: Build a first cylinder ----------------------------------------------------
    z_res = spline_points.shape[0]
    cyl0_dir = spline_points[-1]-spline_points[0]
    cyl0_length = np.linalg.norm(cyl0_dir)
    cyl0_dir = normalize_vector(cyl0_dir)
    cyl0_c = np.mean(spline_points, axis=0)
    
    cyl0 = pv.CylinderStructured(radius=radius, height=cyl0_length, 
                                 center=cyl0_c, direction=cyl0_dir,
                                 theta_resolution=theta_res, z_resolution=z_res)
    if return_intermediates:
        cyl00 = cyl0.copy()
    # Step 2: Translation of circles -----------------------------------------------------
    # Get centers of individual circles
    circle_point_batches_0 = cyl0.points.reshape(z_res,theta_res,3) # reshape points into batches along the cylinder
    cyl0_centers = np.mean(circle_point_batches_0, axis=1)
    # Calculate vectors to corresponding tip spline points
    center_vectors = spline_points - cyl0_centers
    
    # Move points of cylinder so center corresponds to spline point
    cyl0.points = cyl0.points + np.repeat(center_vectors, theta_res, axis=0)
    if return_intermediates:
        cyl01 = cyl0.copy()
    # Step 3: Rotation of circles --------------------------------------------------------
    # Make spline vectors, assuming that they are ordered
    spline_vectors = generate_spline_vectors(spline_points)
    # Find the rotation vectors for each set of circles == the rotation vectors at each spline point
    rot_angles = np.arccos( np.dot(spline_vectors, cyl0_dir) )
    rot_axes = -normalize_vector( np.cross(spline_vectors, cyl0_dir) ) # Minus is needed so rotation is correct
    rot_vectors = ( rot_axes.T*rot_angles ).T
    
    circle_point_batches_1 = cyl0.points.reshape(z_res,theta_res,3)
    for i, (center, rvec) in enumerate( zip(spline_points, rot_vectors) ):
        # Make rotation instance
        r = Rotation.from_rotvec(rvec)
        # Rotate points
        circle_point_batches_1[i] = center + r.apply(circle_point_batches_1[i]-center)
        
    cyl0.points = circle_point_batches_1.reshape(int(z_res*theta_res), 3)
    
    # Turn into PolyData
    cyl_surf = structured_cylinder_to_polydata(cyl0, theta_res, z_res, all_triangles=True)
    # Save some data for further processing
    cyl_surf['circle_centers'] = np.repeat(spline_points, theta_res, axis=0)
    cyl_surf['spline_vectors'] = np.repeat(spline_vectors, theta_res, axis=0)
    
    if return_intermediates:
        # Return intermediates in reverse chronological order
        return cyl_surf, cyl01, cyl00
    return cyl_surf


#%% Move points to desired distance to mid surf

def move_points_to_parallel_surface_pointdist(points, surf, dist, tree_surf=None):
    """Move points to a specified distance to the surface, based on surface points."""
    # Get nearest surface points -> use normal for ray tracing
    if tree_surf == None:
        tree_surf = cKDTree(surf.points)
    # Get distance to surf points
    dist_ps, idx_ps = tree_surf.query(points)
    # Move points to uniform distance from surf
    vec_surf_points = points - surf.points[idx_ps]
    new_points = surf.points[idx_ps] + ( vec_surf_points.T * (dist/dist_ps) ).T
    return new_points


def move_points_to_parallel_surface_raytrace(points, surf, dist, tree_surf=None, 
                                             same_side=True, add_missing_points=True, 
                                             missing_point_method='pointdist'):
    """
    Move points to a surface parallel to the given surface using raytracing along the normals.

    Parameters
    ----------
    points : (N,3), array_like
        Point coordinates.
    surf : pyvista.PolyData mesh
        Surface mesh.
    dist : float
        Target distance to the surface.
    tree_surf : cKDTree, optional
        If the cKDTree of the surface points was already calculated, pass it here. The default is None.
    same_side : bool, optional
        If True, check that all points end up on same side of the surface. The default is True.
    add_missing_points : bool, optional
        If True, add back points for which ray tracing failed. The default is True.
    missing_point_method : str, optional
        'pointdist' or 'original'. If 'pointdist', missing points are moved along the vector to the nearest
        surface point. If 'original' the original missing points are added to the output without
        any shift. The default is 'pointdist'.

    Returns
    -------
    points_moved : (N,3), array_like
        Coordinates of moved points.

    """
    # Get nearest surface points -> use normal for ray tracing
    if tree_surf == None:
        tree_surf = cKDTree(surf.points)
    if 'Normals' not in surf.array_names:
        surf.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    _, idx_ps = tree_surf.query(points)
    ray_dirs = surf['Normals'][idx_ps]
    # Check direction of rays and invert where necessary (to make sure that all points get a ray trace result)
    ray_dir_dot = np.einsum('ij,ij->i', ray_dirs, surf.points[idx_ps]-points)
    ray_dirs[ray_dir_dot < 0] *= -1
   
    # Ray tracing using Pyvista's multi_ray_trace (only available in pyvista 0.27.4, dependencies: rtree, trimesh, pyembree)
    ray_points, ray_idx, ray_cells = surf.multi_ray_trace(points, ray_dirs, first_point=True, retry=True)
    ray_dirs_hit = ray_dirs[ray_idx]
    # Calculate new points
    if same_side:
        if not np.all(ray_dir_dot > 0 ) or np.all(ray_dir_dot < 0 ):
            reference_dir = normalize_vector(np.sum(ray_points - points[ray_idx], axis=0))
            ray_dir_dot_2 = np.dot(ray_dirs_hit, reference_dir)
            ray_dirs_hit[ray_dir_dot_2 < 0] *= -1
            
    points_moved_0 = ray_points - dist*normalize_vector(ray_dirs_hit)
    
    if add_missing_points:
        # Find ids of missing points from
        ids_missing = np.delete( np.arange(points.shape[0]), ray_idx)     
        if missing_point_method == 'pointdist':
            # add points using simple version
            points_moved_1 = move_points_to_parallel_surface_pointdist(points[ids_missing], surf, dist, 
                                                                       tree_surf=tree_surf)
        elif missing_point_method == 'original':
            points_moved_1 = points[ids_missing]
            
        # construct complete array
        points_moved = np.zeros_like(points)
        points_moved[ray_idx, :] = points_moved_0
        points_moved[ids_missing, :] = points_moved_1
    else :
        points_moved = points_moved_0
        
    return points_moved        

# Function to move points to surface, has some more ways to add missing points
def move_points_to_surface_raytrace(points, surf, tree_surf=None, 
                                    add_missing_points=True, 
                                    missing_point_method='pointdist'):
    """
    Move points to a surface along the surface normals.
    
    Compared to the version for parallel surfaces, has more ways to add in missing points.

    Parameters
    ----------
    points : (N,3), array_like
        Point coordinates.
    surf : pyvista.PolyData mesh
        Surface mesh.
    tree_surf : cKDTree, optional
        If the cKDTree of the surface points was already calculated, pass it here. The default is None.
    add_missing_points : bool, optional
        If True, add back points for which ray tracing failed. The default is True.
    missing_point_method : str, optional
        Options are 'pointdist', 'continuous or 'original'. 
        'pointdist': missing points are moved along the vector to the nearest surface point. 
        'continous': Interpolate coordinates of missing points from successful points around them.
        'original': the original missing points are added to the output without any shift. 
        The default is 'pointdist'.

    Returns
    -------
    points_moved : (N,3), array_like
        Coordinates of moved points.

    """
    # Get nearest surface points -> use normal for ray tracing
    if tree_surf == None:
        tree_surf = cKDTree(surf.points)
    if 'Normals' not in surf.array_names:
        surf.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    _, idx_ps = tree_surf.query(points)
    ray_dirs = surf['Normals'][idx_ps]
    # Check direction of rays and invert where necessary (to make sure that all points get a ray trace result)
    ray_dir_dot = np.einsum('ij,ij->i', ray_dirs, surf.points[idx_ps]-points)
    ray_dirs[ray_dir_dot < 0] *= -1
   
    # Ray tracing using Pyvista's multi_ray_trace (only available in pyvista 0.27.4, dependencies: rtree, trimesh, pyembree)
    ray_points, ray_idx, ray_cells = surf.multi_ray_trace(points, ray_dirs, first_point=True, retry=True)
            
    points_moved_0 = ray_points 
    
    if add_missing_points:
        # Find ids of missing points 
        ids_missing = np.delete( np.arange(points.shape[0]), ray_idx)     
        if missing_point_method == 'pointdist':
            # add points using simple version
            points_moved_1 = move_points_to_parallel_surface_pointdist(points[ids_missing], surf, 0, 
                                                                       tree_surf=tree_surf)
        elif missing_point_method == 'continuous':
            # Interpolate values for all points between the successful points
            f_inter = {}
            for i in range(3):
                f_inter[i] = scipy.interpolate.interp1d(ray_idx, ray_points[:,i], kind='cubic')
            points_moved_1 = np.empty((len(ids_missing), 3))
            for j, idx in enumerate(ids_missing):
                if idx > min(ray_idx) and idx < max(ray_idx):
                    for i1 in range(3):
                        points_moved_1[j, i1] = f_inter[i1](idx)
                else :
                    # If it's not in the range of successful ray traces, keep the original point
                    points_moved_1[j] = points[idx]
        elif missing_point_method == 'original':
            points_moved_1 = points[ids_missing]
                        
        # construct complete array
        points_moved = np.zeros_like(points)
        points_moved[ray_idx, :] = points_moved_0
        points_moved[ids_missing, :] = points_moved_1
    else :
        points_moved = points_moved_0
        
    return points_moved

#%% Functions for determining best fron toroid position

def get_ebend_area(mesh, area_params):
    """Get the area of a mesh after cutting at the top and bottom with the given parameters.
    
    area_params is a named tuple with the entries z_rim_min, z_rim_max and curv_dist.
    """
    zmin = area_params.z_rim_min + area_params.curv_dist
    zmax = area_params.z_rim_max - area_params.curv_dist
    clip_tmp = mesh.clip('-z', (0,0,zmin)).clip('z', (0,0,zmax))
    return clip_tmp.area



# Define the function to minimize
def get_fill_area_front_position(x, mid_mesh, front_surf, vec_front_back, back_centroid, 
                                 area_params, missing_area=0, return_full=False):
    """
    For a given position of the front toroid, calculate difference between patch area and missing area.
    
    This function is minimized during the optimization of the position of the front half toroid
    when constructing a reference rim mesh.

    Parameters
    ----------
    x : float
        Displacement of front_surf along vec_front_back. This is the parameter to optimize.
    mid_mesh : pyvista.PolyData mesh
        Patch that is clipped by the moved front_surf. After clipping, the patch area is calculated
        and compared to the missing_area.
    front_surf : pyvista.PolyData mesh
        The front mesh to be moved around, usually a half toroid.
    vec_front_back : (3,) array_like
        The vector along which the front_surf is moved.
    back_centroid : (3,) array_like
        Coordinates of the back mesh centroid.
    area_params : namedtuple
        Parameters for get_ebend_area.
    missing_area : float, optional
        Size of the missing area. The default is 0.
    return_full : bool, optional
        If True, function returns cut patch and displacement vector in addition
        to the area difference. The default is False.

    Returns
    -------
    area_diff : float
        Difference between clipped patch area and missing area.
    
    mid_add_clip : pyvista.PolyData mesh, optional, if return_full
        Clipped patch.
    displacement vector: x*vec_front_back, optional, if return_full

    """
    # Move front surf by vector
    front_surf_tmp = front_surf.copy()
    front_surf_tmp.points += x*vec_front_back

    # Cut the fill area with it
    mid_add_clip = mid_mesh.clip_surface(front_surf_tmp, invert=False) # Default is invert=True!
    mid_centroid = np.mean(mid_mesh.points, axis=0)
    clip_centroid = np.mean(mid_add_clip.points, axis=0)
    if np.linalg.norm((clip_centroid-back_centroid)[:2]) > np.linalg.norm((mid_centroid-back_centroid)[:2]):
        mid_add_clip = mid_mesh.clip_surface(front_surf_tmp, invert=True)
    
    fill_area = get_ebend_area(mid_add_clip, area_params)
    area_diff = abs(fill_area - missing_area)
    if return_full:
        return area_diff, mid_add_clip, x*vec_front_back
    return area_diff


def make_front_surf(front_edge_points):
    """Given an array of front_edge_points, make a mesh going through them using a polynomial surface fit."""
    # Make a nice front surface
    # Problems with surface fit: try expanding front edge points. Points are sorted and alternate between the two sides
    front_edge_mid_points = 0.5* (front_edge_points[0::2]+front_edge_points[1::2])
    front_edge_expanded = front_edge_points + 0.5*(front_edge_points - np.repeat(front_edge_mid_points, 2, axis=0))
    
    front_surf = polynomial_surface_fit_v1(np.r_[front_edge_points, front_edge_expanded], order=3, 
                                           output_step=1, output_extrapol=0.1, align_surf_z=False,return_grid_info=False)
    return front_surf
    
def optimize_toroid_position(patch, front_edge_points, back_edge_points, area_params, missing_area, 
                             x_bounds=None, x_init=0, move_xy=True):
    """
    Optimize the position of the front half toroid while constructing a rim reference mesh.

    Parameters
    ----------
    patch : pyvista.PolyData mesh
        Patch to connect back and front mesh after cutting with a surface through the front_edge_points.
    front_edge_points : (N,3), ndarray
        Points forming the edge of the front half toroid.
    back_edge_points : (N,3), ndarray
        Points forming the edge of the mesh in the back.
    area_params : namedtuple
        Parameters for get_ebend_area.
    missing_area : float
        Size of the missing area.
    x_bounds : None or tuple, optional
        Bounds on x for scipy.optimize.minimize, given as ((min,max),). The default is None.
    x_init : (N,) ndarray, optional
        Initial guess for x. The default is 0.
    move_xy : bool, optional
        If True, restrict movement of the front to x and y, disallowing z movements. The default is True.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Full result of optimization.
    vec_front_back : (3,) ndarray
        Vector along which front is to be moved by res.x.

    """
    print('Calculating best position for the toroid.')
    # Calculate the vector from front to back
    tree_front_edge = cKDTree(front_edge_points)
    _, idx_bf = tree_front_edge.query(back_edge_points)
    if move_xy:
        vec_fb_sum = np.sum(back_edge_points - front_edge_points[idx_bf], axis=0)
        vec_front_back = normalize_vector(np.array([vec_fb_sum[0], vec_fb_sum[1], 0]))
    else:
        vec_front_back = normalize_vector( np.sum(back_edge_points - front_edge_points[idx_bf], axis=0) )
    
    # Make a nice front surface
    front_surf = make_front_surf(front_edge_points)
    # Get the back centroid
    back_centroid = np.mean(back_edge_points, axis=0)
    
    # Find bounds for minimization
    if x_bounds ==None:
        front_edge_plane = fit_plane(front_edge_points, back_centroid, N_to_ref=True)
        plane_distances = np.dot(patch.points - front_edge_plane.center, front_edge_plane.normal)
        x_bounds = ((np.quantile(plane_distances,0.1), np.quantile(plane_distances,0.9)),)
        
    # Run the minimization to find the optimal position of the front half toroid
    t0 = time.time()
    res = minimize(partial(get_fill_area_front_position, mid_mesh=patch, front_surf=front_surf, 
                           vec_front_back=vec_front_back, back_centroid=back_centroid, 
                           area_params=area_params, 
                           missing_area=missing_area, return_full=False), 
                           (x_init), bounds=x_bounds, tol=0.5)
    
    if res.success:
        print('Resulting x: {:.2f}. This took {:.1f} seconds.'.format(res.x[0], time.time()-t0))
    else:
        print('Minimization not successful. This took {:.1f} seconds.'.format(time.time()-t0))
        
    return res, vec_front_back

#%% Functions for stitching meshes together

def point_upwards(vectors):
    """Make all input vectors point upwards (positive z)."""
    vectors[vectors[:,2] < 0] *= -1
    return vectors
    
def find_closest_pointcloud(list_pointclouds, test_points):
    """Given a list of point clouds and an array of test points, find the closest point cloud for each test point."""
    list_distances = []
    for p in list_pointclouds:
        tree_tmp = cKDTree(p)
        dist_tmp, _ = tree_tmp.query(test_points)
        list_distances.append(dist_tmp)
    dist_all = np.c_[list_distances].T
    return np.argmin(dist_all, axis=1)

def find_missing_rows(a,b):
    """For each row in b, find out if it's present in a. Return ids of all rows not present in a."""
    b_in_a = b[:,None]==a
    full_row = np.sum(b_in_a, axis=-1)
    full_b_in_a = np.sum(full_row == b.shape[1], axis=1)
    return np.where(full_b_in_a==0)[0]


def surf_find_new_borders_vertical(surf, old_border_key='border_orig', add_edges=True):
    """After clipping a surface, find its fresh borders where it was cut.
    
    Dependencies: mesh_find_borders, normalize_vector, point_upwards. 
    Assumes that the new borders are vertical.
    Returns dictionary with point ids and points belonging to the new border.
    """
    # Find all border ids, and discard the ones that were already border ids before
    all_border_ids, _, all_border_edges = mesh_find_borders(surf)
    isin_old_border = surf[old_border_key][all_border_ids]
    new_border_ids = all_border_ids[isin_old_border != 1]
    
    # Add back upper / lower edge points if necessary
    if add_edges:
        edges_new_border_verts = np.isin(all_border_edges, new_border_ids) 
        corner_edge_ids = np.where( np.sum(edges_new_border_verts, axis=1) ==1)[0] # gives 2 if completely in new edge, 0 if in old edge, 1 if at border
        # Calculate mean direction of all new border vectors (align to point upwards!) as reference vector
        new_border_edge_ids = np.where( np.sum(edges_new_border_verts, axis=1) ==2)[0]
        new_border_edge_vectors = surf.points[all_border_edges[new_border_edge_ids][:,1]] - surf.points[all_border_edges[new_border_edge_ids][:,0]]
        new_border_mean_direction = normalize_vector( np.mean(point_upwards(new_border_edge_vectors), axis=0) )
        # Calculate direction of corner vectors
        corner_edge_vectors = surf.points[all_border_edges[corner_edge_ids][:,1]] - surf.points[all_border_edges[corner_edge_ids][:,0]]
        corner_edge_vectors = normalize_vector(point_upwards(corner_edge_vectors))
        
        corner_vec_angles = np.rad2deg( np.arccos( np.dot(corner_edge_vectors, new_border_mean_direction) ) )
        corner_edge_outpoint_ids = all_border_edges[corner_edge_ids][np.invert( edges_new_border_verts[corner_edge_ids])]
        assert np.sum(np.isin(corner_edge_outpoint_ids, new_border_ids)) == 0
        
        new_border_ids_all = np.sort( np.r_[new_border_ids, corner_edge_outpoint_ids[corner_vec_angles < 45]] )
        new_border_points_all = surf.points[new_border_ids_all]
        
        # Save for output
        new_border = {'ids': new_border_ids_all, 'points': new_border_points_all}
    else:
        new_border = {'ids': new_border_ids, 'points': surf.points[new_border_ids]}
    
    return new_border
                
def check_and_fix_surface(patch, expected_edges):
    """
    Check whether a patch contains all expected edges, and fix the patch if this is not the case.

    Parameters
    ----------
    patch : pyvista.PolyData mesh
        The mesh to be checked.
    expected_edges : ndarray
        Array of expected edges.

    Returns
    -------
    pyvista.PolyData
        The fixed patch.

    """
    patch_all_edges, patch_edge_counts, patch_faces = mesh_find_edges(patch)
    missing_edge_ids = find_missing_rows(patch_all_edges, expected_edges)
    if len(missing_edge_ids) == 0:
        # If no edges are missing, just return original mesh
        return patch
    missing_edge_point_ids = expected_edges[missing_edge_ids]
    
    # Check if missing points are already connected to the mesh through faces
    missing_points_in_faces = np.isin(missing_edge_point_ids, patch_faces)
    # Separate two cases: simple missing connections, or if vertices are missing completely from the faces
    missing_connections = missing_edge_point_ids[np.sum(missing_points_in_faces, axis=1) == 2]
    missing_vertices = np.unique(missing_edge_point_ids[np.invert(missing_points_in_faces)])
    
    # Add missing connections --------------------------------------------------
    new_faces = []
    # First check: is patch connected, or does it have gaps?
    conn = patch.connectivity()
    point_region_ids = conn.point_arrays['RegionId']
    if len( np.unique(point_region_ids) ) > 1:
        missing_patch_connections = []
        patch_ids = []
        missing_connections_samepatch = []
        for row in missing_connections:
            a = point_region_ids[np.where(conn.points == patch.points[row[0]])[0][0]]
            b = point_region_ids[np.where(conn.points == patch.points[row[1]])[0][0]]
            if a != b:
                missing_patch_connections.append(row)
                patch_ids.append(np.sort(np.array([a,b])))
            else: 
                missing_connections_samepatch.append(row)
        # Process the connections between two patches
        patch_ids = np.array(patch_ids)
        missing_patch_connections = np.array(missing_patch_connections)
        patch_connections_unique, counts = np.unique(patch_ids, axis=0, return_counts=True)
        for i, patch_connection in enumerate(patch_connections_unique):
            if counts[i] != 2:
                # This is probably the case when there's an overhang on ones side
                continue
                #print('Check this again: no two connections between two patches!')
            else:
                connection_point_ids = np.where(np.sum(patch_ids == patch_connection, axis=1)==2)[0]
                point_ids_tmp = missing_patch_connections[connection_point_ids]
                new_faces.append(np.r_[len(point_ids_tmp.flatten()), point_ids_tmp.flatten()])
        # overwrite missing_connections for further processing
        missing_connections = np.array(missing_connections_samepatch)
            
    if len(missing_connections) > 0:
        for (start, end) in missing_connections:
            geo = patch.geodesic(start, end)
            new_faces.append(np.r_[geo.n_points, geo['vtkOriginalPointIds']])
    # Add missing vertices
    if len(missing_vertices) > 0:
        all_face_vertices = np.sort( np.unique(patch_faces.flatten()) )
        for vertex in missing_vertices:
            i1 = np.searchsorted(all_face_vertices, vertex)
            if i1 > np.max(all_face_vertices) or i1 ==0:
                # If it's out, continue
                continue                
            start = all_face_vertices[i1-1]
            end = all_face_vertices[i1]
            geo = patch.geodesic(start, end)
            new_faces.append(np.r_[geo.n_points+1, geo['vtkOriginalPointIds'], vertex])
    if len(new_faces) == 0:
        print('Could not add in missing edges, fix this later!')
        return patch
    patch_new_faces = np.r_[patch.faces, np.concatenate(new_faces, axis=0)]
    patch_new = pv.PolyData(patch.points, patch_new_faces).triangulate()
    return patch_new            

def regular_mesh_find_xy(mesh):
    """Find the size in x and y directions of a regular mesh."""
    # Distance to next point
    dist_next = np.linalg.norm(mesh.points[1:] - mesh.points[:-1], axis=1)
    # Find first occurrence of the high value
    cutoff_tmp = 0.5*(np.max(dist_next) - np.min(dist_next))
    cutoff_id = np.where(dist_next > cutoff_tmp)[0][0]
    size_x = int(cutoff_id + 1)
    assert mesh.n_points % size_x == 0, 'Determination of mid mesh x and y failed!'
    size_y = int(mesh.n_points / size_x)
    return(size_x, size_y)


def decimate_regular_mesh(mesh, size_xy, division_factors=(2,2), list_transfer_arrays=[]):
    """Decimate a regular pyvista mesh by the division factors for (x,y).
    
    This assumes that the mesh points are on a regular grid, 
    points are ordered first along x and then y.
    """
    # Get new ids along the x and y axes
    new_axes = {}
    for i, (key, n) in enumerate( zip(['x','y'], size_xy) ):
        new_axes[key] = np.arange(n)[0::division_factors[i]]
        # Preserve outer edges
        if n-1 not in new_axes[key]:
            new_axes[key] = np.r_[new_axes[key], int(n-1)]
        new_axes[key+'_size'] = len(new_axes[key])
    # Make the new faces
    faces_new = make_2d_grid_faces((new_axes['y_size'], new_axes['x_size']))
    # Extract the points that will be present in the new mesh
    point_ids_all = np.arange(mesh.n_points).reshape(size_xy[1], size_xy[0]).T
    point_ids_new_2d = point_ids_all[new_axes['x'],:][:,new_axes['y']]
    point_ids_new = point_ids_new_2d.T.flatten()
    assert np.all( np.sort(point_ids_new) == point_ids_new)
    # Build the new mesh
    new_mesh = pv.PolyData( mesh.points[point_ids_new], faces_new.flatten() )
    # If desired, transfer properties from old mesh arrays
    if len(list_transfer_arrays) > 0:
        for name in list_transfer_arrays:
            new_mesh[name] = mesh[name][point_ids_new]
            
    return new_mesh

def decimate_regular_mesh_v1(mesh, division_factors=(2,2), list_transfer_arrays=[]):
    """Get xy size and run mesh decimation of regular mesh."""
    size_xy = regular_mesh_find_xy(mesh)
    new_mesh = decimate_regular_mesh(mesh, size_xy, division_factors, list_transfer_arrays)
    return new_mesh
    
    
    
    
    
    
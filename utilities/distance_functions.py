# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:39:35 2022

@author: Anna
"""
import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from collections import namedtuple


from fitting_functions import fit_straight_line
from utility_functions import normalize_vector


#%% Simple distance functions

def distance_along_point_array(x):
    """Cumulative distance along an array of ordered points."""
    dist_stepwise = np.linalg.norm(x[1:] - x[:-1], axis=1)
    dist_stepwise = np.concatenate( (np.zeros(1), dist_stepwise) )
    dist_cumulative = np.cumsum(dist_stepwise)
    return dist_cumulative

#!! output now given as named tuple
def minimum_distance_refined(a,b):
    """
    Determine the minimum distances between two sets of points a and b.
    
    Distance is determined in several steps  
      1. dist_ab: Minimum distances for each point in a to all points in b
      2. Cleaning: check for points in b that were found by more than one point in a. 
          Exclude all of the determined points in a except for the one with the smallest distance.
      3. Determine minimum distances b -> a
          3.1 In the resulting list of points in a, find all that were excluded before and add them back
          3.2 Add the b -> a distance now, taking the smallest value if there are several    


    Parameters
    ----------
    a : numpy.ndarray
        (n,3) array of xyz coordinates of points in group a.
    b : numpy.ndarray
        (n,3) array of xyz coordinates of points in group b.

    Returns
    -------
    results : named tuple
        Named tuple of results, containing the following numpy.ndarrays:
            - dist_refined: refined minimum distances a -> b
            - a_refined: points in a contributing to dist_refined
            - b_refined: points in b contributing to dist_refined
            - orig_dist_ab: original minimum distances a -> b
            - orig_idx_ab: ids of b points closest to a points (original)
            - orig_dist_ba: original minimum distances b -> a
            - orig_idx_ba: ids of a points closest to b points (original)
    """     
    # Determine KDTrees
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    # Find minimum distances a->b
    dist_ab, idx_ab = tree_b.query(a)
    # Check idx_ab for duplicates, giving unique values, the inverse indices for mapping them back, and the counts
    unique, inverse_indices, counts = np.unique(idx_ab, return_inverse=True, return_counts=True) # returns unique indices, indices for mapping them back and counts of these unique indices
    # Find indices in unique where counts are >1
    unique_to_test = np.where(counts>1)[0] 

    points_to_exclude = np.array([]) # initialize array
    
    # Determine the points to exclude
    for idx_unique in unique_to_test: # loop over all indices that were found several times as nearest neighbours
        points_to_test = np.where(inverse_indices==idx_unique)[0] # These are the indices of the points to test
        idx_min = np.argmin(dist_ab[points_to_test]) # Find the one with the smallest distance
        p_exclude = np.delete(points_to_test, idx_min) # Take out the point with the smallest distance
        points_to_exclude = np.concatenate([points_to_exclude, p_exclude]) # All others are added to the list of points to exclude
    
    # Exclude the points from a, dist_ab and idx_ab
    dist_ab_cleaned = np.delete(dist_ab, points_to_exclude.astype(np.int))
    idx_ab_cleaned = np.delete(idx_ab, points_to_exclude.astype(np.int))
    a_cleaned = np.delete(a, points_to_exclude.astype(np.int), axis=0)
    
    # Now find minimum distances b->a
    dist_ba, idx_ba = tree_a.query(b)
    
    dist_add = np.array([]) # initialize array
    idx_a_add = np.array([])
    idx_b_add = np.array([])

    # Check for previously excluded a points that now appear as minimum distance partners of b
    for point in points_to_exclude.astype(np.int): # Loop over previously excluded points
        idx1 = np.where(idx_ba == point)[0].astype(np.int)
        if len(idx1) == 1:
            dist_add = np.append(dist_add, dist_ba[idx1]) # add the new b->a distance
            idx_a_add = np.append(idx_a_add, point) # add the point index to the indices of a points to add
            idx_b_add = np.append(idx_b_add, idx1) # index of b point to add
        elif len(idx1) > 1:
            idx_of_idx1 = np.argmin(dist_ba[idx1]).astype(np.int)
            idx1_min = idx1[idx_of_idx1]
            
            dist_add = np.append(dist_add, dist_ba[idx1_min]) # add the new b->a distance
            idx_a_add = np.append(idx_a_add, point) # add the point index to the indices of a points to add
            idx_b_add = np.append(idx_b_add, idx1_min) # index of b point to add
    
    # Add back the identified points with newly defined distance
    dist_refined = np.concatenate([dist_ab_cleaned, dist_add])
    idx_ab_refined = np.concatenate([idx_ab_cleaned, idx_b_add])
    a_refined = np.concatenate([a_cleaned, a[idx_a_add.astype(np.int)]])
    b_refined = b[idx_ab_refined.astype(np.int)]
    
    # Define a named tuple to store results
    result_tuple = namedtuple('distances', ['dist_refined', 'a_refined', 'b_refined',
                                            'orig_dist_ab', 'orig_idx_ab',
                                            'orig_dist_ba', 'orig_idx_ba'])
    results = result_tuple(dist_refined, a_refined, b_refined, 
                           dist_ab, idx_ab, dist_ba, idx_ba)
    
    return results

def minimum_distance_cleaned(a,b):
    """
    Determine the minimum distances of points in a to points in b.
    
    Corresponds to first two steps of minimmum_distance_refined, leaving out the adding back procedure
    and thus focusing exclusively on the minimum distance of a to b.
      1. dist_ab: Minimum distances for each point in a to all points in b
      2. Cleaning: check for points in b that were found by more than one point in a. 
          Exclude all of the determined points in a except for the one with the smallest distance.

    Parameters
    ----------
    a : numpy.ndarray
        (n,3) array of xyz coordinates of points in group a.
    b : numpy.ndarray
        (n,3) array of xyz coordinates of points in group b.

    Returns
    -------
    results : named tuple
        Named tuple of results, containing the following numpy.ndarrays:
            - dist_cleaned: cleaned minimum distances a -> b
            - a_cleaned: points in a contributing to dist_cleaned
            - b_cleaned: points in b contributing to dist_cleaned
            - idx_ab_cleaned: ids of b points closest to cleaned a points
            - orig_dist_ab: original minimum distances a -> b
            - orig_idx_ab: ids of b points closest to a points (original)
    """     
    # Determine KDTree of b
    tree_b = cKDTree(b)
    # Find minimum distances a->b
    dist_ab, idx_ab = tree_b.query(a)
    # Check idx_ab for duplicates, giving unique values, the inverse indices for mapping them back, and the counts
    unique, inverse_indices, counts = np.unique(idx_ab, return_inverse=True, return_counts=True) # returns unique indices, indices for mapping them back and counts of these unique indices
    # Find indices in unique where counts are >1
    unique_to_test = np.where(counts>1)[0] 

    points_to_exclude = np.array([]) # initialize array
    
    # Determine the points to exclude
    for idx_unique in unique_to_test: # loop over all indices that were found several times as nearest neighbours
        points_to_test = np.where(inverse_indices==idx_unique)[0] # These are the indices of the points to test
        idx_min = np.argmin(dist_ab[points_to_test]) # Find the one with the smallest distance
        p_exclude = np.delete(points_to_test, idx_min) # Take out the point with the smallest distance
        points_to_exclude = np.concatenate([points_to_exclude, p_exclude]) # All others are added to the list of points to exclude
    
    # Exclude the points from a, dist_ab and idx_ab
    dist_ab_cleaned = np.delete(dist_ab, points_to_exclude.astype(np.int))
    idx_ab_cleaned = np.delete(idx_ab, points_to_exclude.astype(np.int))
    a_cleaned = np.delete(a, points_to_exclude.astype(np.int), axis=0)
    b_cleaned = b[idx_ab_cleaned.astype(np.int)]
    
    # Define a named tuple to store results
    result_tuple = namedtuple('distances', ['dist_cleaned', 'a_cleaned', 'b_cleaned', 'idx_ab_cleaned',
                                            'orig_dist_ab', 'orig_idx_ab'])
    results = result_tuple(dist_ab_cleaned, a_cleaned, b_cleaned, idx_ab_cleaned,
                           dist_ab, idx_ab)
    
    return results

#%% Finding distances by ray tracing

#TODO: Adapt the rim analysis script to work with this (more general) function
def ray_trace_through_midsurf(surf0, surf1, mid_surf, reference_vector=[], from_cells=True, cell_position_key='xyz'):
    """
    Find distances between two meshes surf0 and surf1 by using the normals of mid_surf as ray directions.
    
    The Pyvista mesh mid_surf can e.g. be generated by finding all points midway between nearest
    neighbor pairs of surf0 and surf1, and fitting a polynomial surface into these points.

    Parameters
    ----------
    surf0 : pyvista.PolyData
        First mesh for distance measurement
    surf1 : pyvista.PolyData
        Second mesh for distance measurement
    mid_surf : pyvista.PolyData
        Middle surface between the meshes surf0 and surf1. mid_surf['Normals'] are used as 
        ray tracing vectors.
    reference_vector : 1d numpy.ndarray, optional
        Reference vector to indicate direction of ray tracing (p0->p1). If not given, it's calculated 
        from the points of surf0 and surf1.
    from_cells : bool, optional
        Perform ray tracing starting from the cells (from_cells=True) or the points (from_cells=False)
        of surf0. The default is True.
    cell_position_key : str, optional
        If from_cells=True, cell positions are retrieved from surf0[cell_position_key]. The default is 'xyz',
        corresponding to default naming of pycurv-generated meshes.

    Returns
    -------
    ray_trace_results : named tuple
        Named tuple containing source and target points of rays, ray lengths, 
        as well as coordinates and ids of the corresponding mid_surf points.

    """
    # If no reference vector is given, calculate it from means of p1 and p0
    # Reference vector points from side 0 to side 1
    if len(reference_vector) == 0:
        reference_vector = normalize_vector(np.mean(surf1.points, axis=0) - np.mean(surf0.points, axis=0))
    
    # Get the relevant points on side0
    if not from_cells:
        side0_points = surf0.points
    else:
        side0_points = surf0[cell_position_key]
    # For side0: Find closest middle point and take its normal as ray tracing dir
    tree_mid = cKDTree(mid_surf.points)
    dist_s0_m, idx_s0_m = tree_mid.query(side0_points)
    side0_ray_dirs = mid_surf['Normals'][idx_s0_m]
    
    # Check if direction is correct    
    if np.dot(reference_vector, np.mean(side0_ray_dirs, axis=0)) < 0:
        side0_ray_dirs *= -1
    
    # Ray tracing using Pyvista's multi_ray_trace (only available in pyvista 0.27.4, dependencies: rtree, trimesh, pyembree)
    ray_points, ray_idx, ray_cells = surf1.multi_ray_trace(side0_points, side0_ray_dirs, 
                                                           first_point=True, retry=True)
    
    # Get distances
    side0_ray_points = side0_points[ray_idx] # ray_idx = indices of the rays corresponding to the intersection points (ray_points)
    ray_distances = np.linalg.norm( (ray_points - side0_ray_points), axis=1)
    
    # Get corresponding mid points
    ray_mid_ids = idx_s0_m[ray_idx]
    ray_mid_points = mid_surf.points[ray_mid_ids]
    
    # Store results in named tuple
    raytrace_tuple = namedtuple('Ray_tracing_results', ['side0_points', 'side1_points', 'ray_distances', 
                                                        'ray_mid_points', 'ray_mid_ids', 'side0_orig_ids'])
    ray_trace_results = raytrace_tuple(side0_ray_points, ray_points, ray_distances, ray_mid_points, 
                                       ray_mid_ids, ray_idx)    

    return ray_trace_results     


#%% From rim distance functions

def map_points_on_line(points, line_points):
    """
    Map sparse points on a more highly sampled (curved) line based on distance of the points to the line.
    
    Practically, this results in an orthogonal mapping of the points on the line, 
    which is grown until a target point index is found for every line point.

    Parameters
    ----------
    points : numpy.ndarray
        Nx3 or Nx2 Array of points to be mapped on the line.
    line_points : numpy.ndarray
        Mx3 or M2 Array of line points, usually more densely sampled than the target points.

    Returns
    -------
    idx_line_points_new : numpy.ndarray 
        Mx1 Array, same length as line_points. Contains for each line point the idx of the corresponding target point.

    """
    # Generate distance matrix line - points
    distmat_line_points = distance_matrix(line_points, points)    
    # Argsort along axis 0: top row are ids of closest line points for each target point, 
    # 2nd row are ids of second closest line points etc.
    mat_argsort_line_points = np.argsort(distmat_line_points, axis=0)
    # Flatten the array to [row0, row1, row2]
    flat_argsort_line_points = mat_argsort_line_points.flatten()
    # Extract the unique line point ids and their first occurrence in the flattened array
    lineID_unique, lineID_unique_idx = np.unique(flat_argsort_line_points, return_index=True)
    # Make an array full of -1 to fill with correct indices
    idx_line_points_new = np.ones(line_points.shape[0])*-1
    # Fill in the correct target point index for each of the line points ids. 
    # Target point idx is given by (index in flattened array % length of point array)
    idx_line_points_new[lineID_unique] = lineID_unique_idx % points.shape[0]
    
    return idx_line_points_new.astype(int)


def point_dist_splinemapping(query_points, target_points, target_spline):
    """
    Map query on target points fit by a spline.
    
    Map query points on target points using a smooth spline fit through target, 
    and return the distance of each query point to its mapped target point.

    Parameters
    ----------
    query_points : np.ndarray
        Array of query points with shape (Nq,3).
    target_points : np.ndarray
        Array with target points with shape (Nt,3).
    target_spline : np.ndarray
        Spline points of smooth spline fit through target points. The spline is 
        used to determine the mapping of query to target points.

    Returns
    -------
    dist_query_target : np.ndarray of the shape (Nq,)
        Distance of each query point to its mapped target point.
    idx_query_target: np.ndarray of the shape (Nq,)
        ID of corresponding target point for each query point.

    """
    # Map query points on spline (get closest spline point for each query point) 
    tree_spline = cKDTree(target_spline)
    _, idx_query_spline = tree_spline.query(query_points)
    
    # Map target points on spline points based on nearest neighbors tip -> spline
    # Assumes that spline points are oversampled compared to target points
    # One target point can thus be mapped on several spline points.
    idx_spline_target = map_points_on_line(target_points, target_spline)
    
    # Map query points to target points based on (query -> spline) and (target -> spline) mapping
    # idx_spline_target: contains for each spline point the id of its mapped target point.
    # idx_query_spline: contains for each query point the id of its mapped spline point.
    
    idx_query_target = idx_spline_target[idx_query_spline]
    
    # For each ray mid point, calculate distance to tip 
    dist_query_target = np.linalg.norm( target_points[idx_query_target] - query_points, axis=1)
    
    return dist_query_target, idx_query_target


def point_dist_linemapping(query_points, target_points):
    """
    Map query points on target points fit by a straight line.
    
    Map query points on target points using a straight line fit through target points, 
    and return the distance of each query point to its mapped target point.

    Parameters
    ----------
    query_points : np.ndarray
        Array of query points with shape (Nq,3).
    target_points : np.ndarray
        Array with target points with shape (Nt,3).

    Returns
    -------
    dist_query_target : np.ndarray of the shape (Nq,)
        Distance of each query point to its mapped target point.
    idx_query_target: np.ndarray of the shape (Nq,)
        ID of corresponding target point for each query point.

    """
    # Fit straight line into target points
    line_p, line_dir = fit_straight_line(target_points)

    # Get distance along the straight line for target points and query points
    target_dist_along_line = np.dot( (target_points - line_p), line_dir)
    query_dist_along_line = np.dot( (query_points - line_p), line_dir)

    # Subtract each element from each element in the two arrays to find the closest matches
    # tmp arrays are arrays tiled to make subtraction of each element from each element possible
    tmp1 = np.tile(target_dist_along_line, (len(query_dist_along_line),1)).T
    tmp2 = np.tile(query_dist_along_line, (len(target_dist_along_line),1))
    # subtract and find ids with smallest absolute difference
    idx_query_target = np.argmin( abs(tmp1 - tmp2), axis=0)
    
    # Get the original target point for each query point and calculate the distance
    dist_query_target = np.linalg.norm( target_points[idx_query_target] - query_points, axis=1)
    
    return dist_query_target, idx_query_target

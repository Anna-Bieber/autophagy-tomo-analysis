# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:39:35 2022

@author: Anna
"""
import numpy as np
from scipy.spatial import cKDTree
from collections import namedtuple

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
            - orig_idx_ab: ids of a points closest to b points (original)
            - orig_dist_ba: original minimum distances b -> a
            - orig_idx_ba: ids of b points closest to a points (original)

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


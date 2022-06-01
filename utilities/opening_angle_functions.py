#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phagophore opening angle specific functions.

The opening angle is used as a measure for phagophore completeness.
Created on Tue Sep 28 18:05:53 2021

@author: anbieber
"""
import numpy as np
import scipy.spatial
from collections import namedtuple

from utility_functions import normalize_vector, angle_between_vectors
from fitting_functions import (fit_plane, plane_axes, project_points_on_plane, 
                               fit_plane_and_circle, dist_to_plane_signed)




#%% Functions related to angular binning / circle angular ranges

def get_circle_angle(points, circle):
    """Calculate the angle of points with respect to a circle.
    
    Angle 0 is set at random, but relative angles are interpretable.
    """
    # Generate axes of circle plane and calculate coordinates of query points wrt to this plane
    plane_ax0, plane_ax1 = plane_axes(circle.normal)
    points_plane_coords = project_points_on_plane(points, circle.center, plane_ax0, plane_ax1, 
                                                  output_coord_sys='plane')
    
    # Get angles with respect to plane axes
    # Normalize plane coordinates
    points_plane_coords_norm = normalize_vector(points_plane_coords, axis=1)
    # Get angle w.r.t. plane axes, especially ax0
    angles_axes = np.arccos(points_plane_coords_norm)    
    angles_ax0 = angles_axes[:,0]
    angles_ax0[angles_axes[:,1] > 0.5*np.pi] *= -1 

    return angles_ax0

def bin_points_around_circle(points, circle, bin_number):
    """
    Bin a set of points on a circle into angular batches with respect to the circle.

    Parameters
    ----------
    points : (N,3), array_like
        The point coordinates.
    circle : named tuple
        Circle parameters can be accessed with circle.center and circle.normal.
    bin_number : int
        Number of bins into which a full circle is divided.

    Returns
    -------
    angle_result : named tuple
        Binning result containing bin_ids for each point, bin_edges (in rad), as well as unique_ids
        and unique_counts generated from the bin_ids.

    """
    # Calculate the angle of each point
    angles = get_circle_angle(points, circle)
    # Perform binning
    bin_step = 2*np.pi / bin_number
    angle_bins = np.arange(-np.pi, np.pi+bin_step, bin_step)
    angle_bin_ids = np.digitize(angles, bins=angle_bins)
    # Get unique bin ids and counts
    angle_unique_ids, angle_bin_counts = np.unique(angle_bin_ids, return_counts=True)
    # Store results in a named tuple
    angle_binning = namedtuple('angle_bins', ['bin_ids', 'bin_edges', 'unique_ids', 'unique_counts'])
    angle_result = angle_binning(angle_bin_ids, angle_bins, angle_unique_ids, angle_bin_counts)
    return angle_result


def circle_angular_range(angles, return_deg=False):
    """
    Maximum angular range covered by a set of angles.

    Parameters
    ----------
    angles : (N,), array_like or list
        List of angles in radians.
    return_deg : bool, optional
        If True, maximum angular range is given in degrees. The default is True.

    Returns
    -------
    max_range : float
        Maximum angular range in radians or degrees (if return_deg==True).

    """
    angles_diff = scipy.spatial.distance.cdist(angles.reshape((-1,1)), 
                                               angles.reshape((-1,1)))

    angles_diff[angles_diff > np.pi] -=2*np.pi
    max_range = np.max(abs(angles_diff))
    if return_deg:
        return max_range/np.pi*180
    return max_range


#%% Opening angle functions

def analyze_opening_cone(plane_normals, ref_normal):
    """
    Cone refinement for opening angle calculations.
    
    Since the normals of all opening planes should form a cone, the normal of the 
    cone plane or the vector describing the cone center can be used as refined estimations
    of the rim direction.

    Parameters
    ----------
    plane_normals : (N,3) array_like
        Normals of the rim opening planes.
    ref_normal : (N,) array_like
        Reference normal, e.g. rim plane normal.

    Returns
    -------
    d_results : dict
        Results of cone refinement, including new opening angles.

    """
    # The normals are also the endpoints of a cone with apex (0,0,0). -> get the circle describing the opening
    cone_plane, cone_circle = fit_plane_and_circle(plane_normals, ref_normal, N_to_ref=True)
    # Calculate interesting cone parameters
    cone_center_vector = normalize_vector(cone_circle.center)

    cone_half_angle = np.rad2deg(np.arcsin( np.clip(cone_circle.radius, -1, 1)))    
    angle_ref_cone_center = angle_between_vectors(ref_normal, cone_center_vector)
    if np.dot(cone_center_vector, ref_normal) < 0:
        cone_half_angle = 180 - cone_half_angle
        angle_ref_cone_center = 180 - angle_ref_cone_center
        cone_center_vector *= -1
    # Save in results    
    d_results = {'cone_half_angle': cone_half_angle,
                'angle_ref_cone_normal': angle_between_vectors(ref_normal, cone_plane.normal),
                'angle_ref_cone_center': angle_ref_cone_center,
                'cone_circle_rmse': cone_circle.rmse,
                'cone_plane': cone_plane,
                'cone_circle': cone_circle} 
    # Calculate individual angles 
    for name, vec in zip(['cone_angN', 'cone_angC'], [cone_plane.normal, cone_center_vector]):
        angles_tmp = np.rad2deg( np.arccos( np.dot(plane_normals,vec) ) )
        
        d_results['{}_values'.format(name)] = angles_tmp
        d_results['{}_mean'.format(name)] = np.mean( angles_tmp )
        d_results['{}_std'.format(name)] = np.std( angles_tmp )
        d_results['{}_vector'.format(name)] = vec 
    
    return d_results


def phagophore_opening_planes(rim_points, ph_points, dist_cutoff=50, angle_bin_number=36):
    """
    Get planes fit into the mouth of a phagophore for estimating the opening angle.

    Parameters
    ----------
    rim_points : (N,3), array_like
        Coordinates of rim points (roughly segmented).
    ph_points : (N,3), array_like
        Phagophore points to use for fits, e.g. inner membrane points.
    dist_cutoff : float, optional
        Maximum distance to rim until which phagophore points are defined as opening points. The default is 50.
    angle_bin_number : int, optional
        Number of angular bins into which a full circle (2 pi / 360 degrees) would be divided. The default is 36.

    Returns
    -------
    res_dict : dict
        Dictionary containing opening planes as well as rim plane, rim circle, opening points etc.

    """
    # Fit a plane and circle into rim
    ph_centroid = np.mean(ph_points, axis=0)
    rim_plane, rim_circle = fit_plane_and_circle(rim_points, ph_centroid, N_to_ref=False)
    # Get phagophore points close to the rim
    dist_to_rim = abs( dist_to_plane_signed(ph_points, rim_plane) )  
    opening_points = ph_points[dist_to_rim <= dist_cutoff]
    # Divide points into bins depending on circle angle
    angle_bins = bin_points_around_circle(opening_points, rim_circle, angle_bin_number)
    
    # Fit a plane to each set of points
    median_counts = np.median(angle_bins.unique_counts)
    opening_angles = []
    processed_ids = []
    planes = {}
    
    for idx, counts in zip(angle_bins.unique_ids, angle_bins.unique_counts):
        # Don't process if too few points
        if counts < 0.25*median_counts:
            continue
        # Get points
        points_tmp = opening_points[angle_bins.bin_ids == idx]
        # Fit plane
        plane_tmp = fit_plane(points_tmp, reference_point=ph_centroid, 
                              N_to_ref=True)
        
        # Calculate angle of plane_tmp and rim_plane
        angle = angle_between_vectors(rim_circle.normal, plane_tmp.normal, 
                                      normalized=True, degrees=True)
        # Store angles and ids
        opening_angles.append(angle)
        processed_ids.append(idx)
        planes[idx] = plane_tmp
        
    opening_angles = np.array(opening_angles)
    # Calculate angular range used for opening angle calculation
    angles_present = angle_bins.bin_edges[processed_ids]
    circle_max_angrange = circle_angular_range(angles_present, return_deg=True)    
    
    # Collect results in dictionary    
    res_dict = {'rim_plane': rim_plane,
                'rim_circle': rim_circle,
                
                'opening_angle_mean': np.mean(opening_angles),
                'opening_angle_std': np.std(opening_angles),
                'opening_angle_values': opening_angles,
                
                'opening_points': opening_points,
                'opening_angle_bins': angle_bins,
                'angle_bins_processed': np.array(processed_ids),
                'n_patches': len(processed_ids),
                'opening_planes': planes,
                'opening_plane_normals': np.array( [planes[key0].normal for key0 in planes] ),
                'circle_max_ang_range': circle_max_angrange}
    
    return res_dict

    
def phagophore_opening_angle(rim_points, ph_points, dist_cutoff=50, angle_bin_number=36,
                             cone_refinement=True, cone_ref_min_angular_range=90):
    """
    Calculate the opening angle of a phagophore.

    Parameters
    ----------
    rim_points : (N,3), array_like
        Coordinates of rim points (roughly segmented).
    ph_points : (N,3), array_like
        Phagophore points to use for fits, e.g. inner membrane points.
    dist_cutoff : float, optional
        Maximum distance to rim until which phagophore points are defined as opening points. The default is 50.
    angle_bin_number : int, optional
        Number of angular bins into which a full circle (2 pi / 360 degrees) would be divided. The default is 36.
    cone_refinement : bool, optional
        If True, perform cone refinement if angular range is large enough. The default is True.
    cone_ref_min_angular_range : int or float, optional
        Minimum angular range of the rim circle covered by opening points to allow cone refinement, 
        in degrees. The default is 90.

    Returns
    -------
    d_final : dict
        Final mean, std and values of opening angle.
    d0 : dict
        Full results of opening angle calculations.

    """
    # Get opening planes and initial opening angles
    d0 = phagophore_opening_planes(rim_points, ph_points, dist_cutoff=dist_cutoff, angle_bin_number=angle_bin_number)
    # Keys for final angle results
    angle_val_keys = ['mean', 'std', 'values']
    # Cone refinement (only if enough points are present)
    if cone_refinement and d0['circle_max_ang_range'] >= cone_ref_min_angular_range :
        d0.update( analyze_opening_cone(d0['opening_plane_normals'], d0['rim_plane'].normal) )
        # Save correct angles in final dictionary
        d_final = {'roa_'+key: d0['cone_angN_'+key] for key in angle_val_keys}
        d_final['cone_refined'] = True
    else:
        # Save correct angles in final dictionary
        d_final = {'roa_'+key: d0['opening_angle_'+key] for key in angle_val_keys}
        d_final['cone_refined'] = False
    
    return d_final, d0
    
    
    

    
    
    
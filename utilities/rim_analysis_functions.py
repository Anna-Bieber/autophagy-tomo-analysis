#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specific rim analysis functions.

Created on Fri Dec 18 15:23:47 2020
@author: anbieber
"""

import numpy as np
from pyvistaqt import BackgroundPlotter

from collections import namedtuple
import scipy
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from utility_functions import normalize_vector, PCA, get_outlier_points_dbscan
from fitting_functions import fit_curve_spline_oversampled, polynomial_surface_fit_v1, fit_surface_spline
from histogram_functions import discretize_array_overlap
from mesh_functions import extract_mesh_cell_ids
from distance_functions import ray_trace_through_midsurf

#%% Rim analysis specific functions

def separate_mesh_sides_angles(mesh, subset_ids, 
                               key_coords='xyz', key_normals='normal',
                               angle_cutoff=90, plot_angles=False):
    """
    Separate two sides of points from a mesh based on the surface normal angles.
    
    Mesh should be pyvista PolyData with an array containing coordinates and normals. 
    A plane is fitted into the points and the angle between 
    the plane normal and surface normals of each point is evaluated.

    Parameters
    ----------
    mesh : Pyvista PolyData
        Pyvista mesh with arrays, arrays should contain point coordinates and normals.
    subset_ids : numpy.ndarray
        1d array of indices indicating which subset of points/values of the mesh should be used.
    key_coords : string, optional
        Array name to get point/cell coordinates from mesh. The default is 'xyz'.
    key_normals : string, optional
        Array name to get surface normals from mesh. The default is 'normal'.
    angle_cutoff : int/float or tuple of ints/floats, optional
        Angle cutoff to separate points into sides. 
        Give one value or tuple of (lower, upper) with angles(side0) < lower and angles(side1) > upper. 
        The default is 90.
    plot_angles : True or False, optional
        Should histogram of angles be plotted? The default is False.

    Returns
    -------
    ids_side0 : numpy.ndarray
        1d array of indices applicable to mesh to extract parameters of side 0.
    ids_side1 : numpy.ndarray
        1d array of indices applicable to mesh to extract parameters of side 1.
    points_side0 : numpy.ndarray
        2d array of points of side 0.
    points_side1 : numpy.ndarray
        2d array of points of side 1.

    """
    # Get the relevant points from the mesh
    points = mesh[key_coords][subset_ids]
    # Plane fit to points to get normal
    plane_center, (eigvec, eigval) = PCA(points)
    plane_normal = eigvec[:,-1] / np.linalg.norm(eigvec[:,-1])
    # Calculate angles between cell normals and overall plane normal
    normals = mesh[key_normals][subset_ids]
    angles = np.rad2deg( np.arccos( np.dot(normals, plane_normal) ) )
    
    if plot_angles:
        fig, ax = plt.subplots()
        ax.hist(angles)
        ax.set_xlabel('Angle to plane normal (deg)')
    
    # Apply cutoff angle(s). If only one is given, turn into tuple (min, max) first
    if isinstance(angle_cutoff, (int, float)):
        angle_cutoff = (angle_cutoff, angle_cutoff)
            
    ids_side0 = subset_ids[angles < angle_cutoff[0]]
    ids_side1 = subset_ids[angles > angle_cutoff[1]]
    # Also extract points
    points_side0 = mesh[key_coords][ids_side0]
    points_side1 = mesh[key_coords][ids_side1]
    
    return (ids_side0, ids_side1, points_side0, points_side1)



def make_middle_surface(a, b, fit_type='polynomial', pol_order=3, output_step=1, 
                        output_extrapol=None, align_surf_z=False):
    """
    Generate a surface between two point clouds a and b, works best if a and b are mostly flat and parallel.

    Parameters
    ----------
    a : numpy.ndarray
        2d nx3 array with points of side a.
    b : numpy.ndarray
        2d nx3 array with points of side b.
    fit_type : string
        'polynomial' or 'spline'
    pol_order : int, optional
        Order of polynomial surface fit (if fitting a polynomial surface). The default is 3.
    output_step : int or float, optional
        Spacing of output surface mesh points. The default is 1.
    output_extrapol : float or `None`, optional
        How much should output surface go beyond original points in all directions, 
        as fraction of original dimension. The default is None.
    align_surf_z : True or False, optional
        Whether output surface should be aligned so 2 of the edges are orthogonal to z

    Returns
    -------
    middle_surf : Pyvista mesh
        Pyvista mesh containing the determined middle surface points,
        and an array 'Normals' containing point-wise surface normals.
    grid_info : named tuple
        Additional information on the output mesh: array of ids in correct shape, edge ids and edge point coordinates.

    """
    # Generate cKDTrees for both sides
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    
    # Sample with points from other side
    dist_ab, idx_ab = tree_b.query(a)
    dist_ba, idx_ba = tree_a.query(b)
    
    # Get middle points between the nearest neighbors
    middle_points_ab = ( a + b[idx_ab] ) * 0.5
    middle_points_ba = ( b + a[idx_ba] ) * 0.5
    middle_points_combined = np.concatenate((middle_points_ab, middle_points_ba), axis=0)

    # Fit surface to middle points
    if fit_type == 'polynomial':
        middle_surf, grid_info = polynomial_surface_fit_v1(middle_points_combined, order=pol_order, 
                                             output_step=output_step, output_extrapol=output_extrapol,
                                             align_surf_z=align_surf_z, return_grid_info=True)
    elif fit_type == 'spline':
        middle_surf, grid_info = fit_surface_spline(middle_points_combined, 
                                                    output_step=output_step, 
                                                    output_extrapol=output_extrapol,
                                                    align_surf_z=align_surf_z, 
                                                    return_grid_info=True)
    
    return middle_surf, grid_info


    
def extract_points_along_surface(points, normals, surf_points, surf_normals, 
                                 bin_axis=2, bin_width=1, min_bin_counts=10,
                                 smooth=True, spline_s=500):
    """
    Extract points of a mesh where it is orthogonal to a second mesh.
    
    Given two surfaces, extract points in the first surface 
    whose normals are most orthogonal to the closest point 
    normals in the second surface. This function was created to extract the tip 
    points of a phagophore rim given the middle surface through the phagophore.

    Parameters
    ----------
    points : numpy.ndarray
        Points in query surface (e.g. phagophore tip region).
    normals : numpy.ndarray
        Surface normals of points.
    surf_points : numpy.ndarray
        Points of the probing surface (e.g. the middle surface through the phagophore).
    surf_normals : numpy.ndarray
        Normals of surf_points.
    bin_axis : int, optional
        In which axis direction should points be binned to generate initial result? 0=x, 1=y, 2=z. The default is 2.
    bin_width : float or int, optional
        Bin width for initial tip point generation. The default is 1.
    smooth : True or False, optional
        Should initially determined points be smoothed and resampled by spline fitting? The default is True.
    spline_s : float, optional
        Smoothing factor for spline fitting. The default is 500.

    Returns
    -------
    res1 : numpy.ndarray
        Array of refined result points after spline smoothing & resampling.
    res1_idx : numpy.ndarray
        Indices to find res1 within input points: points[res1_idx] = res1
    res0 : numpy.ndarray
        Array of original result points.
    res0_idx : numpy.ndarray
        Indices to find res0 within input points: points[res0_idx] = res0

    """
    # Check if points are sorted along bin_axis, sort if this is not the case
    re_sorted=False
    if np.any( (points[:, bin_axis][1:]-points[:, bin_axis][:-1]) < 0):
        re_sorted = True
        print('Sorting points along bin axis..')
        sort_ids = np.argsort(points[:,bin_axis])
        points = points[sort_ids]
        normals = normals[sort_ids]
    # Calculate boundary ids for binning of points along bin_axis
    bin_start = (points[0,2] // bin_width)*bin_width # Bin start should be first relevant multiple of bin_width
    boundary_ids = discretize_array_overlap(points[:,2],bin_width=bin_width, 
                                            overlap=0, bin_start=bin_start)
    # NEW 2021/05/01: Problem: sometimes bins just contain one point, which messes up the tip determination
    # Use a minimum bin size and merge bins if they have less counts
    boundary_ids = np.array(boundary_ids)
    bin_counts = boundary_ids[:,1]-boundary_ids[:,0]
    too_empty_bin_ids = np.where(bin_counts < min_bin_counts)[0]
    if len(too_empty_bin_ids) > 0:
        # Put bins into previous bins except if it's the first bin. This assumes that there are no two consecutive too empty bins
        for bin_id in too_empty_bin_ids:
            if bin_id == 0:
                boundary_ids[1,0] = boundary_ids[0,0]
            else:
                boundary_ids[bin_id-1,1] = boundary_ids[bin_id, 1]
        # Delete too empty bins
        boundary_ids = np.delete(boundary_ids, too_empty_bin_ids, axis=0)
    # For each point (sorted), get closest surface points
    tree_surf = cKDTree(surf_points)
    _, idx_ps = tree_surf.query(points) # Use sorted points - needed for binning later
    
    # Calculate dot products between corresponding normals
    dot_normals = np.einsum('ij,ij->i', normals, surf_normals[idx_ps,:])
    
    # Go through bins and find point with dot product closest to 0
    res0_idx = []
    for bounds in boundary_ids: # boundary ids correspond to sorted high_k1 points
        # Extract dot products of this slice
        dot_tmp = dot_normals[slice(*bounds)]
        # Get index of value closest to 0
        idx_dot0 = np.argmin(abs(dot_tmp))
        # Calculate index of corresponding point within sorted array and add this point to the output list
        res0_idx.append(bounds[0] + idx_dot0)
    
    res0_idx = np.array(res0_idx)
    res0 = points[res0_idx, :]
    if re_sorted:
        res0_idx = sort_ids[res0_idx] # If points were re-sorted, return original indices
    
    # If desired, make spline smoothing of first result and retrieve closest points
    if smooth:
        # Spline fit of res0 -> generate oversampled array of spline points
        spline_point_array = fit_curve_spline_oversampled(res0, smooth_factor=spline_s, oversample_factor=5)

        # Make KDTree of tip region points and use to find nearest neighbors of spline points
        tree_points = cKDTree(points)
        _, idx_spline_p = tree_points.query(spline_point_array)
        
        # Get refined tip 
        res1_idx = np.unique(idx_spline_p)
        res1 = points[res1_idx]
        
        if re_sorted:
            res1_idx = sort_ids[res1_idx] # If points were re-sorted, return original indices
        
        return res1, res1_idx, res0, res0_idx, spline_point_array # Also return original points & spline point array for now
    
    else:
        return res0, res0_idx


def separate_sides_phagophore_rim(curv_mesh, k1_limit_initial=0.08, pix_size=1.408, tip_spline_smooth=500,
                                  plot_sides_initial=True, plot_sides_final=True, clean_sides_clustering=False,
                                  mid_surf_pol_order=3):
    """
    Separate a phagophore rim pycurv mesh into tip and side points and generate a middle surface.

    Parameters
    ----------
    curv_mesh : pyvista.PolyData mesh
        Mesh of the rim membrane with curvature values saved as associated arrays.
    k1_limit_initial : float, optional
        Kappa 1 cutoff for initial separation. The default is 0.08.
    pix_size : float, optional
        Pixel size [nm] with which data were taken, used to determine bin size internally. The default is 1.408.
    tip_spline_smooth : float, optional
        Smoothing factor for spline fit through tip points. The default is 500.
    plot_sides_initial : bool, optional
        If True, initially separated sides are plotted in a pyvistaqt BackgroundPlotter. The default is True.
    plot_sides_final : bool, optional
        If True, final separation is plotted in a pyvistaqt BackgroundPlotter. The default is True.
    clean_sides_clustering : bool, optional
        If True, the two sides are cleaned with DBSCAN clustering to get rid of single misplaced points. The default is False.
    mid_surf_pol_order : int, optional
        Polynomial order for middle surface fitting. The default is 3.

    Returns
    -------
    tip : named tuple 
        (ids, points) tuple of ids and points classified as tip.
    side0 : named tuple 
        (ids, points) tuple of ids and points classified as side0.
    side1 : named tuple 
        (ids, points) tuple of ids and points classified as side1.
    mid_surf : pyvista.PolyData mesh
        Middle surface between side0 and side1, saved as pyvista PolyData mesh.
    sides_extra : named tuple
        Intermediate results of the side separation.        
    p2 : pyvistaqt.BackgroundPlotter, optional, if plot_sides_final=True
        If plot_sides_final==True, this is the BackgroundPlotter instance containing the plot.       
    mid_surf_info : named tuple
        Additional information on middle surface: array of ids in correct shape, 
        edge ids and edge point coordinates.
    """
    # PART 1: Initial separation to get middle surface --------------------------------------
    
    # Use kappa1 limit to get tip region & points purely on sides
    ids_high_k1 = np.where(curv_mesh['kappa_1'] > k1_limit_initial)[0]
    points_high_k1 = curv_mesh['xyz'][ids_high_k1]
    ids_low_k1 = np.where(curv_mesh['kappa_1'] < k1_limit_initial)[0]
    
    # Divide into sides based on angles (mean plane vs surface normals)
    side_sep0 = separate_mesh_sides_angles(curv_mesh, ids_low_k1, 
                                           key_coords='xyz', key_normals='normal', 
                                           angle_cutoff=90, plot_angles=False)
    (ids_s0_tmp, ids_s1_tmp, points_s0_tmp, points_s1_tmp) = side_sep0
    
    # Check separation
    if plot_sides_initial:
        p1 = BackgroundPlotter()
        p1.enable_eye_dome_lighting()
        p1.add_mesh(points_s0_tmp, color='blue', label='side_0')
        p1.add_mesh(points_s1_tmp, color='green', label='side_1')
        p1.add_text('Initial separation')
    
    # Get intial middle surface
    mid_surf0, _ = make_middle_surface(points_s0_tmp, points_s1_tmp, 
                                       fit_type='polynomial', pol_order=mid_surf_pol_order, output_step=1, 
                                       align_surf_z=True)
    
    if plot_sides_initial:
        p1.add_mesh(mid_surf0, color='yellow', label='mid_surf')
        p1.add_legend()
        
    # PART 2: Get refined tip points using middle surface ----------------------------------

    # Sort high_k1 points by z value (needed for binning later)
    high_k1_sorted = {}
    high_k1_sorted['sorting_idx'] = np.argsort(points_high_k1[:,2])
    high_k1_sorted['points'] = points_high_k1[high_k1_sorted['sorting_idx']] 
    high_k1_sorted['orig_ids'] = ids_high_k1[ high_k1_sorted['sorting_idx'] ]
    high_k1_sorted['normal'] = curv_mesh['normal'][ high_k1_sorted['orig_ids'] ]
    
    # Extract smoothed tip points
    tip_results = extract_points_along_surface(high_k1_sorted['points'], high_k1_sorted['normal'],
                                               mid_surf0.points, mid_surf0['Normals'],
                                               bin_axis=2, bin_width=pix_size, min_bin_counts=10,
                                               smooth=True, spline_s=tip_spline_smooth)
    # Unpack results and calculate original ids of tip points
    (tip_points, tip_ids_local, tip_rough_points, tip_rough_ids_local, tip_spline) = tip_results
    tip_ids = high_k1_sorted['orig_ids'][tip_ids_local]
    tip_rough_ids = high_k1_sorted['orig_ids'][tip_rough_ids_local]
    
    # Named tuple for storing point/id results
    mesh_subset_tuple = namedtuple('mesh_subset', ['ids', 'points'])
    tip = mesh_subset_tuple(tip_ids, tip_points)
    
    # PART 3: Final division of all points except the tip ----------------------------------
    
    # Divide the tip region points except for the tip points themselves into the two sides
    # Mean vector side0 -> side1 (if the direction is known, also mean mid surf normal could be used)
    vec_side_01 = normalize_vector( np.mean(points_s1_tmp, axis=0) - np.mean(points_s0_tmp, axis=0) )
    
    # Make KDTree of tip points
    tree_tip = cKDTree(tip.points)
    
    # Get tip region points
    tip_region_keys = ['points', 'orig_ids']
    tip_region = {}
    for key in tip_region_keys:
        tip_region[key] = np.delete(high_k1_sorted[key], tip_ids_local, axis=0)
    
    # Get closest tip points and calculate vector tip -> tip_region_points
    _, idx_points_tip = tree_tip.query(tip_region['points'])
    tip_region['Vec_from_tip'] = normalize_vector(tip_region['points'] - tip.points[idx_points_tip,:])
    
    # Calculate dot product 
    tip_region['Tip_dot_vec01'] = np.dot(tip_region['Vec_from_tip'], vec_side_01)
    
    # Extract orig ids of both sides
    tip_region['side0_orig_ids'] = tip_region['orig_ids'][ tip_region['Tip_dot_vec01'] < 0 ]
    tip_region['side1_orig_ids'] = tip_region['orig_ids'][ tip_region['Tip_dot_vec01'] > 0 ]
    
    # Combine the old and new side ids to get complete side separation
    side0_ids_0 = np.sort( np.concatenate((ids_s0_tmp, tip_region['side0_orig_ids'])) )
    side1_ids_0 = np.sort( np.concatenate((ids_s1_tmp, tip_region['side1_orig_ids'])) )
    
    # If desired, clean up sides using clustering
    if clean_sides_clustering:
        # Get ids of outliers for each side
        outlier_ids ={}
        for i, side_ids in enumerate([side0_ids_0, side1_ids_0]):
            # Get points
            side_points_tmp = curv_mesh['xyz'][ side_ids ]
            # Get ids of outlier points
            outlier_ids[i] = get_outlier_points_dbscan(side_points_tmp)
            # If it's empty, turn into an empty list since empty arrays can't be used for indexing
            if outlier_ids[i].size == 0:
                outlier_ids[i] = []
            
        # Delete outliers from each side and add them to the other side
        side0_ids = np.concatenate((np.delete(side0_ids_0, outlier_ids[0]), side1_ids_0[outlier_ids[1]]))
        side1_ids = np.concatenate((np.delete(side1_ids_0, outlier_ids[1]), side0_ids_0[outlier_ids[0]]))
        print('Moving {} points from side 0 to side 1, and {} points from side 1 to side 0.'.format(len(outlier_ids[0]), len(outlier_ids[1])))
    else:
        side0_ids = side0_ids_0
        side1_ids = side1_ids_0
        
    # Save together with points as mesh subset
    side0 = mesh_subset_tuple(side0_ids, curv_mesh['xyz'][ side0_ids ])
    side1 = mesh_subset_tuple(side1_ids, curv_mesh['xyz'][ side1_ids ])
    
    # Make a new middle surface using completed sides
    mid_surf, mid_surf_info = make_middle_surface(side0.points, side1.points, 
                                                  fit_type='polynomial', pol_order=mid_surf_pol_order, 
                                                  output_step=pix_size / 4,
                                                  align_surf_z=True)
    
    # Save some extra output as named tuple
    side_sep_tuple = namedtuple('side_sep', ['tip_spline', 'tree_tip', 'tip_rough_ids', 'tip_region', 'vec_side_01'])
    sides_extra = side_sep_tuple(tip_spline, tree_tip, tip_rough_ids, tip_region, vec_side_01)
    
    # Plot final separation
    if plot_sides_final:
        p2 = BackgroundPlotter()
        p2.enable_eye_dome_lighting()
        p2.add_mesh(side0.points, color='blue', label='side_0')
        p2.add_mesh(side1.points, color='green', label='side_1')
        p2.add_mesh(tip.points, color='red', label='tip')
        
        p2.add_mesh(mid_surf, color='white', label='mid_surf')
        #p2.add_arrows(mid_surf.points, mid_surf['Normals'], color='white', label='mid_surf_normals')
        
        p2.add_text('Final separation')
        p2.add_legend()
        
        return tip, side0, side1, mid_surf, sides_extra, p2, mid_surf_info
    
    return tip, side0, side1, mid_surf, sides_extra, mid_surf_info

#%% Ray tracing related functions


def rim_ray_tracing(rim_mesh, mid_surf, side0_ids, side1_ids):
    """
    Determine intermembrane distances at the rim by ray tracing.
    
    Uses normals of mid_surf for ray tracing. Point ids of side0 and side1 are needed 
    to divide the full rim mesh into two separate meshes.

    Parameters
    ----------
    rim_mesh : pyvista.PolyData
        Mesh of the rim membrane.
    mid_surf : pyvista.PolyData
        Mesh of the middle surface: Smooth surface in the middle between the two sides of the rim.
    side0_ids : numpy.ndarray
        (N,1) array of indices of all cells belonging on side0, applicable to rim_mesh.points.
    side1_ids : TYPE
        (N,1) array of indices of all cells belonging on side1, applicable to rim_mesh.points..

    Returns
    -------
    ray_trace_results : named tuple
        Results of ray tracing. Named tuple containing source and target points of rays, ray lengths, 
        as well as coordinates and ids of the corresponding mid_surf points.

    """
    # Make meshes of side0 and side1 (extract from full mesh)
    surf0 = extract_mesh_cell_ids(rim_mesh, side0_ids, keep_cells=True)
    surf1 = extract_mesh_cell_ids(rim_mesh, side1_ids, keep_cells=True)
    
    # Perform ray tracing
    ray_trace_results = ray_trace_through_midsurf(surf0, surf1, mid_surf, from_cells=True)
    
    return ray_trace_results


def get_extrema_rim_hist(H1, back_cutoff_id, bins_dist, bin_step_dist, verbose=True):
    """
    Get the global extrema of a 2D histogram of values measured for a phagophore rim.

    Parameters
    ----------
    H1 : numpy masked array
        The 2D histogram of values.
    back_cutoff_id : int
        Index at which the "back" part of the rim begins.
    bins_dist : numpy.ndarray
        Distance edge values of bins in the rows of the histogram.
    bin_step_dist : float
        Distance binning step size along rows of the histogram.
    verbose : bool, optional
        The default is True.

    Returns
    -------
    global_extrema : dict
        Ids, values and positions of global extrema along the rows.

    """
    # Define results dictionary
    global_extrema = {'max': {}, 'min': {}}
    # Get maximum positions and maximum values
    global_extrema['max']['ids'] = np.argmax(H1, axis=1)
    global_extrema['max']['values'] = np.max(H1, axis=1)
    
    # For minimum: mask out values before max and maybe after back_cutoff_ID (think about this)
    H2 = H1.copy()
    for i, max_ID in enumerate(global_extrema['max']['ids']):
        H2[i,0:max_ID] = np.ma.masked
    H2[:, back_cutoff_id:] = np.ma.masked
        
    # Get minimum positions and minimum values
    global_extrema['min']['ids'] = np.argmin(H2, axis=1)
    global_extrema['min']['values'] = np.min(H2, axis=1)   
    
    for key in ['max', 'min']:
        vals = global_extrema[key]['values']
        if verbose:
            print('{} intermembrane distance: mean {:.2f}, std {:.2f}.'.format(key, np.mean(vals), np.std(vals)))
        global_extrema[key]['imdist_mean'] = np.mean(vals)
        global_extrema[key]['imdist_std'] = np.std(vals)
        
        # Get positions of extrema
        global_extrema[key]['pos_nm'] = bins_dist[global_extrema[key]['ids']] + 0.5*bin_step_dist
        pos = global_extrema[key]['pos_nm']
        if verbose:
            print('{} distance from tip: mean {:.2f} nm, std {:.2f} nm.'.format(key, np.mean(pos), np.std(pos)))
        
        global_extrema[key]['pos_nm_mean'] = np.mean(pos)
        global_extrema[key]['pos_nm_std'] = np.std(pos)
        
    return global_extrema

#%% Curvature / bending energy functions

# Bending energy:
    # Integral dA of 2*kappa*(M-m)^2
    # M : mean curvature
    # m : spontaneous curvature
    # rigidity kappa: 10 k_B T
    # k_B : get with scipy.constants.Boltzmann
    # T: scipy.constants.zero_Celsius + 30 (yeast grown at 30°C)

def bending_energy(area, M, m=0, T_celsius=30, K_factor=10):
    """Calculate the Helfrich bending energy for a membrane mesh with given areas and mean energies (M).
    
    Assumes a bending rigidity of 10 k_B T. m is the spontaneous curvature (default is zero).
    """
    K = K_factor*scipy.constants.Boltzmann*(T_celsius + scipy.constants.zero_Celsius)
    E_local = 2*K*np.multiply(area, np.power((M-m),2))
    E_total = np.sum(E_local)
    return E_total, E_local

def bending_energy_half_cylinder(L, r, T_celsius=30, K_factor=10):
    """Helfrich bending energy for a half cylinder.
    
    Assuming a rigidity of 10 k_B T and a spontaneous curvature of 0.
    """
    K = K_factor*scipy.constants.Boltzmann*(T_celsius + scipy.constants.zero_Celsius)
    E = 0.5*K * L * np.pi / r
    return E

#%% From rim_distance_functions

def align_mid_array_to_tip(mid_surf_info, tip_points):
    """
    Align a 2d array of mid surf indices along a tip.
    
    The output array has the id of the mid_surf corner point closest to the highest 
    tip point at position [0,0], and the id of the mid surf corner point closest 
    to the lowest tip point at position [-1,0].

    Parameters
    ----------
    mid_surf_info : named tuple containing fields ('id_array', 'edge_ids', 'edge_points')
        id array is a 2d array containing all indices of the mid_surf, shaped in the form of the mid surf
        edge_ids and edge_points: ids/points of corners of id array
    tip_points : np.ndarray
        tip points of a phagophore rim.

    Returns
    -------
    mid_surf_array : 2d np.ndarray
        Aligned array of mid surf ids.

    """
    # As reference: get highest and lowest tip point
    tip_sortz_ids = np.argsort(tip_points[:,2])
    tip_lowest = tip_points[tip_sortz_ids[0] ]
    tip_highest = tip_points[tip_sortz_ids[-1] ]
    
    # Find out which of the edge points is the upper front one and which is the lower front one
    dist_upfront = np.linalg.norm(mid_surf_info.edge_points - tip_highest, axis=1)
    edge_id_upfront = mid_surf_info.edge_ids[np.argmin(dist_upfront)]
    dist_lowfront = np.linalg.norm(mid_surf_info.edge_points - tip_lowest, axis=1)
    edge_id_lowfront = mid_surf_info.edge_ids[np.argmin(dist_lowfront)]
    
    # Get corners of original mid id array clock-wise starting from (0,0)
    array_corners = [(0,0), (0,-1), (-1,-1), (-1,0)]
    corner_ids_orig_clockwise = np.array([mid_surf_info.id_array[a] for a in array_corners])
    # Calculate number of 90 deg rotations until edge_id_upfront is (0,0)
    n_rot = np.where(corner_ids_orig_clockwise == edge_id_upfront)[0]
    
    # Rotate array
    mid_surf_array = np.rot90(mid_surf_info.id_array, k=n_rot, axes=(0,1))
    
    # Make sure that edge_id_lowfront is at (-1,0)
    if mid_surf_array[-1,0] != edge_id_lowfront:
        mid_surf_array = mid_surf_array.T
    
    return mid_surf_array

def tipdist_difference_geo_direct_v1(mid_surf, mid_surf_info, tip_points, tree_tip):
    """
    Difference between geodesic and direct tip distance for points in the middle surface of a phagophore rim.
    
    For a regular mesh (the phagophore rim middle surface), calculate for each 
    point the difference between the geodesic and the direct distance to the
    corresponding tip point (same z).

    Parameters
    ----------
    mid_surf : Pyvista PolyData mesh, 
        Phagophore rim middle surface. Should be a regular and rectangular mesh and not orthogonal to z.
    mid_surf_info : named tuple containing fields ('id_array', 'edge_ids', 'edge_points')
        id array is a 2d array containing all indices of the mid_surf, shaped in the form of the mid surf
        edge_ids and edge_points: ids/points of corners of id array
    tip_points : np.ndarray
        tip points of a phagophore rim.
    tree_tip : cKDTree
        cKDTree of tip points to determine position of tip.

    Returns
    -------
    dist_diff : numpy.ndarray
        Contains for each of mid_surf.points the difference between the 
        geodesic and the direct distance to the corresponding tip point.
        Corresponding point is taken in same z to avoid artifacts from 
        mesh gridding.
    mid_surf_ids : numpy.ndarray of same shape as mid_surf mesh
        Array with the same shape as mid_surf, containing for each position 
        the ID of the mid_surf.point sitting there. [0,0] is up_front.

    """
    # Get an array with ids of the surface
    mid_surf_ids = align_mid_array_to_tip(mid_surf_info, tip_points)
    
    # Get distances to the tip
    dist_mid_tip, _ = tree_tip.query(mid_surf.points)
    # Transform distances into array shape
    dist_mt_array = dist_mid_tip[mid_surf_ids]
    # For each row, get position of point closest to tip
    tip_pos_per_row = np.argmin(dist_mt_array, axis=1)
                          
    # Prepare empty arrays
    dist_tip_direct = np.empty(mid_surf.n_points)
    dist_tip_geo = np.empty(mid_surf.n_points)
    
    # Go through mesh row by row
    for id_row, tip_pos in zip(mid_surf_ids, tip_pos_per_row):
        # Get coordinates of points in the row & tip point
        points = mid_surf.points[id_row,:]
        tip_point = points[tip_pos, :]
        # Direct distance to tip for each point
        dist_tip_direct[id_row] = np.linalg.norm(points - tip_point, axis=1)
        # Geodesic distance: calculate distances to neighbors, then get cumulative sum
        dist_to_next = np.concatenate( ( np.zeros(1), np.linalg.norm( points[1:] - points[:-1], axis=1 ) ) )
        dist_front_geo = np.cumsum(dist_to_next)
        dist_tip_geo[id_row] = abs(dist_front_geo - dist_front_geo[tip_pos])
    
    # Calculate difference
    dist_diff = dist_tip_geo - dist_tip_direct
    
    return dist_diff, mid_surf_ids













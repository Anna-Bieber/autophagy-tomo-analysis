#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specific functions for analyzing phagophore - vacuole contact sites.

Abbreviations: imdist = intermembrane distance.

@author: anbieber
"""
import numpy as np
import scipy.stats
import scipy.spatial
from collections import namedtuple

from fitting_functions import circle_fit_3d_points
from rim_analysis_functions import bending_energy
from histogram_functions import histogram_1d_statistics

def peak_at_border(dist_to_border, peak_labels, peak_labels_unique, dist_cutoff=10):
    """Check if a peak is at the border of a mesh.
    
    See notebook Run_vacuole_contact_analysis for usage.
    """
    border_peak = []
    for label in peak_labels_unique:
        min_dist_border = np.min(dist_to_border[peak_labels == label])
        border_peak.append(True) if min_dist_border <= dist_cutoff else border_peak.append(False)
    return border_peak


def analyze_ph_vac_peak(mesh, base_values, pix_size=1.408):
    """
    Analyze a peak in the phagophore membrane at a contact site to the vacuole.

    Parameters
    ----------
    mesh : pyvista.PolyData mesh
        A mesh of the peak. Should have cell arrays ['area', 'imdist_nm', 'xyz', 'vac_dist_pix', 'mean_curvature_VV'].
        Assumes that coordinates are in pixels, with size given by pix_size.
    base_values : named tuple
        Named tuple with values for the base mesh, including the fields [imdist, E_bend, area].
    pix_size : float, optional
        Pixel size in nm. The default is 1.408.

    Returns
    -------
    named tuple
        Results of peak analysis, including peak dimensions, distances and difference in 
        bending energy compared to an area of equal size in the base mesh.

    """   
    # Peak surface area
    peak_area = np.sum(mesh['area'])
    # Height of peak = maximum inter-membrane distance - base intermembrane distance
    peak_top_id = np.argmax(mesh['imdist_nm'])
    peak_height = mesh['imdist_nm'][peak_top_id] - base_values.imdist
    
    # Width of peak
    # Get points at half height (half maximum)
    hm_points = mesh['xyz'][abs(mesh['imdist_nm'] - base_values.imdist - 0.5*peak_height) < 0.25*pix_size]
    # Get radius through circle fit
    hm_circle = circle_fit_3d_points(hm_points)
    peak_fwhm = 2*hm_circle.radius
    
    # Vacuole distance: minimum, value at imdist peak and distance of min to imdist peak
    vac_min_dist = np.min(mesh['vac_dist_pix'])*pix_size
    vac_dist_at_peak = mesh['vac_dist_pix'][peak_top_id]*pix_size
    dist_top_to_closest = pix_size*np.linalg.norm( mesh['xyz'][peak_top_id] - mesh['xyz'][np.argmin(mesh['vac_dist_pix'])])
    
    # Correlation imdist - vacdist
    (dist_pearson_corr, dist_pearson_p) = scipy.stats.pearsonr(mesh['imdist_nm'],mesh['vac_dist_pix']*pix_size)
    (dist_spearman_corr, dist_spearman_p) = scipy.stats.spearmanr(mesh['imdist_nm'],mesh['vac_dist_pix']*pix_size)
    
    # Bending energy
    E_bend_peak, _ = bending_energy(mesh['area'], mesh['mean_curvature_VV'])
    E_bend_diff = E_bend_peak - (base_values.E_bend / base_values.area * peak_area)
    kBT = scipy.constants.Boltzmann*(30 + scipy.constants.zero_Celsius)
    E_bend_diff_kBT = E_bend_diff / kBT
    
    # Define named tuple for results
    tuple_results = namedtuple('peak_results', ('area', 'height', 'width', 
                                                'vac_dist_min_nm', 'vac_dist_peak_nm', 'dist_top_closest',
                                                'dist_corr_pearson', 'dist_corr_spearman', 
                                                'E_bend_peak', 'E_bend_diff', 'E_bend_diff_kBT'))
    
    return tuple_results(peak_area, peak_height, peak_fwhm, 
                         vac_min_dist, vac_dist_at_peak, dist_top_to_closest, 
                         dist_pearson_corr, dist_spearman_corr, E_bend_peak, E_bend_diff, E_bend_diff_kBT)
    
def analyze_base_mesh(mesh, pix_size=1.408):
    """
    Analyze a phagophore mesh without peaks to get base values for peak analysis.

    Parameters
    ----------
    mesh : pyvista.PolyData mesh
        Mesh of the phagophore membrane after cutting out peaks.
        Should have cell arrays ['area', 'imdist_nm', 'xyz', 'vac_dist_pix', 'mean_curvature_VV'].
    pix_size : float, optional
        Pixel size in nm. The default is 1.408.

    Returns
    -------
    base_res : named tuple
        Results of base mesh analysis including area, distances and bending energy.

    """
    # Correlation imdist - vacdist
    (dist_pearson_corr, dist_pearson_p) = scipy.stats.pearsonr(mesh['imdist_nm'],mesh['vac_dist_pix']*pix_size)
    (dist_spearman_corr, dist_spearman_p) = scipy.stats.spearmanr(mesh['imdist_nm'],mesh['vac_dist_pix']*pix_size)
    # Bending energy
    E_bend, _ = bending_energy(mesh['area'], mesh['mean_curvature_VV'])
    # Define named tuple for results
    tuple_base = namedtuple('base_values', ('area','imdist', 'vac_dist', 'dist_corr_pearson', 'dist_corr_spearman','E_bend'))
    base_res = tuple_base(np.sum(mesh['area']), np.mean(mesh['imdist_nm']), np.mean(mesh['vac_dist_pix'])*pix_size, 
                          dist_pearson_corr, dist_spearman_corr, E_bend)
    return base_res


#%% Functions for plotting maps

#%% Plotting 2d slices - see below for usage

def plot_slice_xz(data_xy, data_z, ax, slice_val=0, slice_thickness=1, slice_steps=1, **plot_kwargs):
    """Plot an xz slice of 3d data."""
    # Get the x and y values of the relevant points
    slice_ids = np.where(abs(data_xy[:,1]-slice_val) <= 0.5*slice_thickness)[0]
    x_tmp = data_xy[slice_ids,0]
    z_tmp = data_z[slice_ids]
    
    H, bin_edges, bin_ids = histogram_1d_statistics(x_tmp, z_tmp, slice_steps)
    bin_middles = 0.5*(bin_edges[:-1]+bin_edges[1:])
    ax.fill_between(bin_middles, H['mean']-H['std'], H['mean']+H['std'], alpha=0.5, **plot_kwargs)
    ax.plot(bin_middles, H['mean'], **plot_kwargs)

def plot_slice_yz(data_xy, data_z, ax, slice_val=0, slice_thickness=1, slice_steps=1, **plot_kwargs):
    """Plot an yz slice of 3d data."""
    # Get the x and y values of the relevant points
    slice_ids = np.where(abs(data_xy[:,0]-slice_val) <= 0.5*slice_thickness)[0]
    y_tmp = data_xy[slice_ids,1]
    z_tmp = data_z[slice_ids]
    
    H, bin_edges, bin_ids = histogram_1d_statistics(y_tmp, z_tmp, slice_steps)
    bin_middles = 0.5*(bin_edges[:-1]+bin_edges[1:])
    ax.fill_betweenx(bin_middles, H['mean']-H['std'], H['mean']+H['std'], alpha=0.5, **plot_kwargs)
    ax.plot(H['mean'], bin_middles, **plot_kwargs)

def plot_slices(data_xy, data_z, slice_point, ax_xz, ax_yz, slice_args, ax_limits=None):
    """Add an xz and yz slice to a plot with given axes."""
    ax = ax_xz
    plot_slice_xz(data_xy, data_z, ax, slice_val=slice_point[1], **slice_args)
    if type(ax_limits) in [list, tuple]:
        ax.set_xlim(ax_limits)
    ax.set_xticklabels([])

    ax = ax_yz
    plot_slice_yz(data_xy, data_z, ax, slice_val=slice_point[0], **slice_args)
    if type(ax_limits) in [list, tuple]:
        ax.set_ylim(ax_limits)
    ax.set_yticklabels([])

def add_slice_lines(slice_point, slice_thickness, map_ax, color='black', icon_ax=None, s=2):
    """Add slicing lines into map."""
    # Plot the lines in the map, taking into account the slice thickness
    map_line_args = {'linestyle':'--', 'linewidth': 0.2, 'color': color}
    ax = map_ax
    ax.axvline(slice_point[0]-0.5*slice_thickness, **map_line_args)
    ax.axvline(slice_point[0]+0.5*slice_thickness, **map_line_args)
    ax.axhline(slice_point[1]-0.5*slice_thickness, **map_line_args)
    ax.axhline(slice_point[1]+0.5*slice_thickness, **map_line_args)
    # If wanted, plot the lines and point into the icon ax as well
    if icon_ax is not None:
        ax = icon_ax
        ax.set_xlim(map_ax.get_xlim())
        ax.set_ylim(map_ax.get_ylim())
        ax.scatter(slice_point[0], slice_point[1], color=color, s=s)
        ax.axvline(slice_point[0], color=color)
        ax.axhline(slice_point[1], color=color)
        ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

#%% Functions for plotting convex hulls

def plot_convex_hull(hull, points, ax, **plot_kwargs):
    """Make a line plot of a convex hull."""
    hull_closed = np.r_[hull.vertices, hull.vertices[0]]
    ax.plot(points[hull_closed,0], points[hull_closed,1], **plot_kwargs)
            

def fill_convex_hull(hull, points, ax, col):
    """Plot a filled convex hull."""
    # Get points and ids of min x and max x points
    hull_points = points[hull.vertices]
    id_xmin = np.argmin(hull_points[:,0])
    id_xmax = np.argmax(hull_points[:,0])
    max_id = int(len(hull_points)-1)
    
    # Get sequences of ids for moving along the upper or lower part of the convex hull
    # Since the convex hull moves counterclockwise, the min->max trace is always the lower one
    if id_xmax > id_xmin:
        ids_lower = np.arange(id_xmin, id_xmax+1).astype(int)
        ids_upper = np.flip( np.r_[np.arange(id_xmax, max_id+1), np.arange(0,id_xmin+1)].astype(int) )
    else:
        ids_lower = np.r_[np.arange(id_xmin, max_id+1), np.arange(0,id_xmax+1)].astype(int)
        ids_upper = np.flip( np.arange(id_xmax, id_xmin+1).astype(int) )
    # Get the ordered points and the function describing the middle line
    points_lower = hull_points[ids_lower]
    points_upper = hull_points[ids_upper]
    
    def linear_function(p0,p1):
        m = (p1[1]-p0[1]) / (p1[0]-p0[0])
        a = p0[1]-m*p0[0]
        return lambda x: m*x+a
    f_lin = linear_function(points_lower[0], points_lower[-1])
    # Fill
    ax.fill_between(points_lower[:,0], points_lower[:,1], f_lin(points_lower[:,0]), color=col)
    ax.fill_between(points_upper[:,0], points_upper[:,1], f_lin(points_upper[:,0]), color=col)

        
# Plot maps using convex hulls 
def map_imdist_vacdist(ax, plane_points, imdist, vacdist, imdist_steps, imdist_cmap, vacdist_steps, vacdist_cmap,
                      ax_limits=None):
    """Plot imdist as filled convex hull and vacdist as lines on top."""
    # Make the convex hulls and plot them for imdist:
    for i,z in enumerate(imdist_steps):
        if np.max(imdist) < z:
            continue
        points = plane_points[imdist >= z]
        hull = scipy.spatial.ConvexHull(points)
        fill_convex_hull(hull, points, ax, col=imdist_cmap(i))#, edgecolor=None)
    # Make the convex hulls and plot them for vacdist:
    for i,z in enumerate(vacdist_steps):
        if np.max(vacdist) < z or np.min(vacdist) > z:
            continue
        points = plane_points[vacdist <= z]
        hull = scipy.spatial.ConvexHull(points)
        plot_convex_hull(hull, points, ax, color=vacdist_cmap(i))
    # If axis limits are given, set them
    if type(ax_limits) in [list, tuple]:
        ax.set_xlim(ax_limits)
        ax.set_ylim(ax_limits)        

# Final function
def plot_map_and_slices_gridspec(fig, gs, plane_points, dist0, dist1, ax_limits, 
                                 slice_args= {'slice_thickness': 2, 'slice_steps': 1, 'color': 'black'} ):
    """Make a complete map with slices within a gridspec."""
    # Make the axes
    gs1 = gs.subgridspec(6,6)
    # Plot the map
    ax0 = fig.add_subplot(gs1[1:,:-1])
    map_imdist_vacdist(ax0, plane_points, dist0['values'], dist1['values'], 
                       dist0['steps'], dist0['cmap'], dist1['steps'], dist1['cmap'],
                       ax_limits=ax_limits)
    # Plot the slices
    pmax = plane_points[np.argmax(dist0['values'])]
    ax1 = fig.add_subplot(gs1[0,:-1])
    ax2 = fig.add_subplot(gs1[1:,-1])
    plot_slices(plane_points, dist0['values'], pmax, ax1, ax2, slice_args, ax_limits)
    
    # Add the slice lines
    ax3 = fig.add_subplot(gs1[0,-1])
    add_slice_lines(pmax, slice_args['slice_thickness'], ax0, color='black', icon_ax=ax3, s=3)
    return gs1, [ax0, ax1, ax2, ax3]    
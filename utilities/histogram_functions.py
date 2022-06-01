# -*- coding: utf-8 -*-
"""
Functions for binning, making histograms, finding peaks in histograms etc.

@author: Anna
"""
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats
import scipy.signal

from utility_functions import gcd_floats, replace_nans_with_mean, find_successive_ones



#%% Binning functions

def discretize_array_idlist(x, bin_width=1, bin_start=None, return_edges=False):
    """
    Return edge indices for a sorted array and a given bin width, no overlap.
    
    Lower edge value <= bin < Upper edge value
    
    Parameters
    ----------
    x : np.ndarray
        Input 1D array, sorted in ascending order.
    bin_width : float or int, optional
        Default is 1.
    bin_start : float or None, optional
        Start value of the binning. If `None`, the minimum value of x is taken. Default is `None`.
    return_edges : True or False
        If True, return an array "edge_values" containing the values of the bin edges
    Returns
    -------
    boundary_ids : List of tuples 
        Boundary ids (idx_lower, idx_upper) for dividing x into bins.
    edge_values : numpy.ndarray
        Array of bin edge values
    """
    if bin_start is None:
        bin_start = x.min()
        
    edge_values = np.arange(bin_start, x.max()+bin_width, bin_width)
    boundary_ids = []
    idx_lower = np.searchsorted(x, edge_values[0], side='left')
    for val_upper in edge_values[1:]:
        idx_upper = np.searchsorted(x, val_upper, side="left")
        boundary_ids.append( (idx_lower, idx_upper) )
        
        idx_lower = idx_upper
    
    if return_edges:
        return boundary_ids, edge_values
    
    return boundary_ids

def group_ids_to_window(ids, window_size=1, step=1):
    """
    Group input binning indices into groups with a certain window & step size and return new boundary indices.

    Parameters
    ----------
    ids : list
        List of tuples of indices. `[(idx_lower, idx_upper), ...]`
    window_size : int, optional
        Window size. The default is 1.
    step : int, optional
        Step size with which the window is moved along the id list. The default is 1.

    Returns
    -------
    boundary_ids : list
        Upper and lower indices for each window. `[(idx_lower, idx_upper), ...]`.
    """
    window = deque(maxlen=window_size)

    boundary_ids = []
    for i, value in enumerate(ids):
        window.append(value)
        if len(window) < window.maxlen: continue # wait till window is full        
        if  (i-window_size+1) % step: continue # skip in between steps

        idx_lower = window[0][0]
        idx_upper = window[-1][1]
        
        boundary_ids.append( (idx_lower, idx_upper) )
    
    return boundary_ids

def discretize_array_overlap(x, bin_width=1, overlap=0, bin_start=None):
    """
    Divide a sorted array into bins with a given width and overlap, and return the edge indices.
    
    Bin definition: Lower edge value <= bin < Upper edge value
    
    Parameters
    ----------
    x : np.ndarray
        Input 1D array, sorted in ascending order.
    bin_width : float or int, optional
        Default is 1.
    overlap : float or int, optional
        Overlap of bins, given in the same unit as the bin_width.
    bin_start : float or None, optional
        Start value of the binning. If `None`, the minimum value of x is taken. Default is `None`.

    Returns
    -------
    boundary_ids : List of tuples (idx_lower, idx_upper)
    """
    if overlap == 0:
        boundary_ids = discretize_array_idlist(x, bin_width=bin_width, bin_start=bin_start, return_edges=False)
    else :
        # tmp bin is the greatest common divisor of bin_width & overlap
        bin_tmp = gcd_floats(bin_width, overlap) 
        # Generate non-overlapping ids
        ids_tmp = discretize_array_idlist(x, bin_width=bin_tmp, bin_start=bin_start, return_edges=False)
        # Calculate windowsize & step size as multiples of tmp bin width
        window_size = int(bin_width / bin_tmp)
        step = int(overlap / bin_tmp)
        boundary_ids = group_ids_to_window(ids_tmp, window_size=window_size, step=step)
        
    return boundary_ids

#%% 1D histograms

def histogram_1d_values(a, values, bin_steps, return_bins=False):
    """
    Generate a 1D histogram of values binned over a.

    Parameters
    ----------
    a : numpy.ndarray
        Coordinates for binning (e.g. position, time...).
    values : numpy.ndarray
        Values associated with a, to be displayed in the histogram.
    bin_steps : int or float
        Step size for binning of a.
    return_bins : bool, optional
        If True, returns bins of a. The default is False.

    Returns
    -------
    H : numpy.ma.masked_array
        Histogram of values binned over a. All bins with zero counts are masked.
    H_counts : numpy.ndarray
        Value counts for each bin.
    extent : list
        Boundaries of binned parameter (a) [lower_boundary, higher_boundary].
    bins_a : numpy.ndarray, optional if return_bins=True
        Array of all bin edges.  
    """
    # Calculate bins
    bins_a = np.arange(a.min(), a.max()+bin_steps, step=bin_steps)

    # Generate histograms with counts and summed values:
    H_counts = np.histogram(a, bins=bins_a)[0]
    H_weights = np.histogram(a, bins=bins_a, weights=values)[0]
    
    # Calculate histogram with means and mask out counts==0
    mask = H_counts != 0 # Mask contains everything where counts !=0
    H = np.divide(H_weights, H_counts, where=mask)
    H = np.ma.array(H, mask=~mask)
    # Calculate extent
    extent = [bins_a[0], bins_a[-1]]
    
    if return_bins:
        return H, H_counts, extent, bins_a
    
    return H, H_counts, extent 


def histogram_1d_statistics(a, values, bin_steps, statistics='basic'):
    """
    Generate a 1D histogram with statistics for values binned over a.
    
    Uses scipy.stats.binned_statistic internally. Options for statistics: 
        'basic': mean, std, count or 
        'full': same as basic + median, min, max.
        
    Parameters
    ----------
    a : numpy.ndarray
        Coordinates for binning (e.g. position, time...).
    values : numpy.ndarray
        Values associated with a for which histogram statistics are calculated.
    bin_steps : int or float
        Step size for binning of a.
    statistics : string, optional
        Either 'basic' or 'full'. 
        If 'basic', ther returned statistics are mean, standard deviation and count.
        If 'full', output additionally contains median, min and max.
        The default is 'basic'.

    Returns
    -------
    H : dict
        Dictionary of different histogram arrays with the given statistics.    
    bin_edges : array
        Array of bin edges.
    bin_ids : array
        Indices of bins in which each value of a belongs.    
    """
    # Calculate bins
    bins_a = np.arange(a.min(), a.max()+bin_steps, step=bin_steps)
    # Use scipy.stats.binned_statistic to calculate different statistics in bins
    if statistics == 'basic':
        eval_stats = ['mean', 'std', 'count']
    elif statistics == 'full':
        eval_stats = ['mean', 'std', 'count', 'median', 'min', 'max']
    
    H = {}
    for s in eval_stats:
        H[s], bin_edges, bin_ids = scipy.stats.binned_statistic(a, values, statistic=s, bins=bins_a)
        
    return H, bin_edges, bin_ids
        
def histogram_1d_polydata(mesh, a_name, value_names, bin_steps, return_bins=False):
    """
    Generate 1d histograms of data associated with a mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh with associated values.
    a_name : string
        key for accessing values for binning.
    value_names : string or list of strings
        keys for data for calculating histogram values based on binning of a.
    bin_steps : float
        Binning step size.
    return_bins : bool, optional
        If true, array of bin edges is returned. The default is False.

    Returns
    -------
    H : dict of masked arrays
        For each value_name, contains the histogram of values binned over a. 
        All bins with zero counts are masked.
    H_counts : numpy.ndarray
        Value counts for each bin.
    extent : list
        Boundaries of binned parameter (a) [lower_boundary, higher_boundary].
    bins_a : numpy.ndarray, optional if return_bins=True
        Array of all bin edges.  

    """    
    # Get array values to bin in histogram
    a = mesh[a_name]
    
    # Calculate bins
    bins_a = np.arange(a.min(), a.max()+bin_steps, step=bin_steps)    
    # Calculate extent
    extent = [bins_a[0], bins_a[-1]]
    
    # Generate histograms with counts and make a mask
    H_counts = np.histogram(a, bins=bins_a)[0]
    mask = H_counts != 0 # Mask contains everything where counts !=0
    
    # Generate value histograms
    H = {}
    if type(value_names) != list:
        value_names = [value_names]
    
    for n in value_names:
        print('Making histogram for {}'.format(n))
        # Calculate histogram with means and mask out counts==0
        H_weights = np.histogram(a, bins=bins_a, weights=mesh[n])[0]
        
        H[n] = np.divide(H_weights, H_counts, where=mask)
        H[n] = np.ma.array(H[n], mask=~mask)
    
    if return_bins:
        return H, H_counts, extent, bins_a
    
    return H, H_counts, extent 

# Plot 1d histograms
def plot_hist1d_std(x, hist_dict, xlabel=None, ylabel=None, title=None, ax=None, dpi=150):
    """Plot a 1d histogram as line, with a grey area indicating the standard deviation of the values in each bin.
    
    Parameters
    ----------
    x : x values of bins (1d array)
    hist_dict : dictionary with the keys 'mean' and 'std' indicating the mean and std values for each bin.
    xlabel, ylabel, title: indicate if desired, otherwise none are written (default None).
    ax: pyplot axis or None: 
        If ax is given, the histogram is plotted into this ax and no object is returned. 
        If ax=None, the function returns fig, ax
    dpi: only relevant if ax=None, default is 150.
    
    Returns
    -------
    fig, ax if ax was set to None.
    
    """
    if ax == None:
        fig, ax = plt.subplots(dpi=dpi)
        new_fig = True
    else:
        new_fig = False
    ax.fill_between(x, hist_dict['mean']-hist_dict['std'], 
                    hist_dict['mean']+hist_dict['std'], color='#a9a9a9', label='plus/minus std')
    ax.plot(x, hist_dict['mean'], label='mean')
    
    ax.legend()
    
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
        
    if new_fig:
        return fig, ax
    
    
#%% 2D Histogram functions

def histogram_2d_values(x, y, values, bin_steps, return_bins=False, return_transposed=False):
    """
    Generate a 2d histogram of values binned over x and y.

    Parameters
    ----------
    x : numpy.ndarray
        Values for binning in x.
    y : numpy.ndarray
        Values for binning in y.
    values : string or list of strings
        Data for calculating histogram values based on binning of x and y.
    bin_steps : [float, float] or (float,float)
        Binning step sizes for x and y.
    return_bins : bool, optional
        If true, the two arrays of bin edges are returned. The default is False.
    return_transposed : bool, optional
        If true, histogram and counts are transposed. The default is False.

    Returns
    -------
    H : masked array
        2D histogram of values binned over x and y. 
        All bins with zero counts are masked.
    H_counts : numpy.ndarray
        Value counts for each bin.
    extent : list
        Boundaries of binned parameters x and y [x_lower, x_higher, y_lower, y_higher].
    bins_x : numpy.ndarray, optional, if return_bins=True
        Array of all bin edges in x.  
    bins_y : numpy.ndarray, optional, if return_bins=True
        Array of all bin edges in y. 
    """    
    # Calculate bins
    bins_x = np.arange(x.min(), x.max()+bin_steps[0], step=bin_steps[0])
    bins_y = np.arange(y.min(), y.max()+bin_steps[1], step=bin_steps[1])
    # Generate histograms with counts and summed values:
    H_counts = np.histogram2d(x, y, bins=[bins_x, bins_y])[0]
    H_weights = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=values)[0]
    # Calculate histogram with means and mask out counts==0
    mask = H_counts != 0 # Mask contains everything where counts !=0
    H = np.divide(H_weights, H_counts, where=mask)
    H = np.ma.array(H, mask=~mask)
    # Calculate extent
    extent = [bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]]
    # Return transposed array? Can be plotted immediately with imshow
    if return_transposed:
        H = H.T
        H_counts = H_counts.T
    
    if return_bins:
        return H, H_counts, extent, bins_x, bins_y
    
    return H, H_counts, extent 

def histogram_2d_polydata(mesh, x_name, y_name, value_names, 
                          bin_steps, return_bins=False, return_transposed=False):
    """
    Generate 2d histograms of data associated with a mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh with associated values.
    x_name : string
        key for accessing values for binning in x.
    y_name : string
        key for accessing values for binning in y.
    value_names : string or list of strings
        keys for data for calculating histogram values based on binning of x and y.
    bin_steps : [float, float] or (float,float)
        Binning step sizes for x and y.
    return_bins : bool, optional
        If true, the two arrays of bin edges are returned. The default is False.
    return_transposed : bool, optional
        If true, histogram and counts are transposed. The default is False.

    Returns
    -------
    H : dict of masked arrays
        For each value_name, contains the 2D histogram of values binned over x and y. 
        All bins with zero counts are masked.
    H_counts : numpy.ndarray
        Value counts for each bin.
    extent : list
        Boundaries of binned parameters x and y [x_lower, x_higher, y_lower, y_higher].
    bins_x : numpy.ndarray, optional, if return_bins=True
        Array of all bin edges in x.  
    bins_y : numpy.ndarray, optional, if return_bins=True
        Array of all bin edges in y. 
    """    
    # Get x and y values
    x = mesh[x_name]
    y = mesh[y_name]
    
    # Calculate bins
    bins_x = np.arange(x.min(), x.max()+bin_steps[0], step=bin_steps[0])
    bins_y = np.arange(y.min(), y.max()+bin_steps[1], step=bin_steps[1])
    # Calculate extent
    extent = [bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]]
    
    # Generate histograms with counts and make a mask
    H_counts = np.histogram2d(x, y, bins=[bins_x, bins_y])[0]
    mask = H_counts != 0 # Mask contains everything where counts !=0
    
    # Generate value histograms
    H = {}
    if type(value_names) != list:
        value_names = [value_names]
    
    for n in value_names:
        # print('Making histogram for {}'.format(n))
        # Calculate histogram with means and mask out counts==0
        H_weights = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=mesh[n])[0]
        
        H[n] = np.divide(H_weights, H_counts, where=mask)
        H[n] = np.ma.array(H[n], mask=~mask)
        
        # Return transposed array? Can be plotted immediately with imshow
        if return_transposed:
            H[n] = H[n].T
            
    # Return transposed array? Can be plotted immediately with imshow
    if return_transposed:
        H_counts = H_counts.T
    
    if return_bins:
        return H, H_counts, extent, bins_x, bins_y
    
    return H, H_counts, extent 

#%% Find peaks in the rows of a 2D histogram  

def get_peaks_histogram_rows_v2(H, SG_win_len=9, SG_pol_order=2, peak_dist=7.04, 
                                peak_prom=1.408, peak_width=5, n_cut_empty_bins=0):
    """
    Find peaks in the rows of a 2D histogram.
    
    The data in the rows are first filtered with a Savitzky-Golay filter, then peaks are
    detected using scipy.signal.find_peaks.

    Parameters
    ----------
    H : numpy.ndarray or np.ma.masked_array
        2D histogram to analyze peaks.
    SG_win_len : int, optional
        Window length for Savitzky-Golay filter, corresponds to window_length in scipy.signal.savgol_filter. 
        The default is 9.
    SG_pol_order : int, optional
        Polynomial order for Savitzky-Golay filter, corresponds to polyorder in scipy.signal.savgol_filter. 
        Must be less than SG_win_len. The default is 2.
    peak_dist : float, optional
        Minimal distance between neighbouring peaks, corresponds to distance in scipy.signal.find_peaks. 
        If two peaks are closer than peak_dist, the larger one is retained. The default is 7.04.
    peak_prom : float, optional
        Required peak prominence, corresponds to prominence in scipy.signal.find_peaks. 
        The default is 1.408.
    peak_width : float, optional
        Required peak width, corresponds to width in scipy.signal.find_peaks.
        The default is 5.
    n_cut_empty_bins : int, optional
        If > 0, find stretches of empty bins with the given size and cut the rows afterwards
        before peak determination. The default is 0.

    Returns
    -------
    Peaks : dict
        Dictionary of peak finding results. For each row, only the first max and min peak are saved here.
    peak_properties : dict
        Full peak finding results for each row.
    n_peaks : dict
        Number of max and min peaks for each row.

    """
    # Prepare dicts for results
    Peaks = {}
    peak_keys = ['max_1', 'min_1']
    attribute_keys = ['row_ids', 'col_ids', 'prominences', 'widths', 'values_smooth', 'values_orig']
    for key in peak_keys:
        Peaks[key] = {}
        for key1 in attribute_keys:
            Peaks[key][key1] = []
        
    peak_properties = {}
    n_peaks = {'max': [], 'min': []}

    # Determine if and where to cut rows (look for stretches of empty bins)
    if n_cut_empty_bins == 0:
        cut_ids = (np.ones(H.shape[0])*H.shape[1]).astype(int)
    else:
        cut_ids = find_successive_ones(H.mask, n_cut_empty_bins, axis=1)
           
    # Iterate through rows of histogram and get peaks
    for i, values in enumerate(H):
        # Preprocessing: get rid of areas with too many nans and replace other nans with mean
        val_preprocess = replace_nans_with_mean(values[:cut_ids[i]])
        # Smoothing
        val_smooth = scipy.signal.savgol_filter(val_preprocess, window_length=SG_win_len, polyorder=SG_pol_order)
        # Find maxima
        max_peaks, max_properties = scipy.signal.find_peaks(val_smooth[:-SG_win_len], height=0, distance=peak_dist, 
                                                   prominence=peak_prom, width=peak_width, rel_height=0.5)
        # Find minima
        min_peaks, min_properties = scipy.signal.find_peaks(-val_smooth[:-SG_win_len], height=-50, distance=peak_dist, 
                                                   prominence=peak_prom, width=peak_width, rel_height=0.5)
        min_properties['peak_heights'] *= -1 # change sign back for min peak heights
        
        # Store results
        if len(max_peaks) > 0:
            Peaks['max_1']['row_ids'].append(i)
            Peaks['max_1']['col_ids'].append(max_peaks[0])
            Peaks['max_1']['prominences'].append(max_properties['prominences'][0])
            Peaks['max_1']['widths'].append(max_properties['widths'][0])
            Peaks['max_1']['values_smooth'].append(val_smooth[max_peaks[0]])
            Peaks['max_1']['values_orig'].append(values[max_peaks[0]])
    
        if len(min_peaks) > 0:
            Peaks['min_1']['row_ids'].append(i)
            Peaks['min_1']['col_ids'].append(min_peaks[0])
            Peaks['min_1']['prominences'].append(min_properties['prominences'][0])
            Peaks['min_1']['widths'].append(min_properties['widths'][0])
            Peaks['min_1']['values_smooth'].append(val_smooth[min_peaks[0]])
            Peaks['min_1']['values_orig'].append(values[min_peaks[0]])
        
        # Store full peak properties
        peak_properties[i] = {'max_peak_ids': max_peaks,
                              'max_peak_props': max_properties,
                              'min_peak_ids': min_peaks,
                              'min_peak_props': min_properties}
        n_peaks['max'].append(len(max_peaks))
        n_peaks['min'].append(len(min_peaks))
    
    # Turn value lists into np.arrays
    for key in peak_keys:
        for val_key in list(Peaks[key].keys()):
            if type(Peaks[key][val_key]) == list:
                Peaks[key][val_key] = np.array(Peaks[key][val_key])
    
    return Peaks, peak_properties, n_peaks

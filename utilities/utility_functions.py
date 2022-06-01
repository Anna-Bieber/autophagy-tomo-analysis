#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 22:27:48 2021

Utility functions.

@author: anbieber
"""
import numpy as np
from sklearn.cluster import DBSCAN
import scipy.signal 
import scipy.ndimage
from pyvistaqt import BackgroundPlotter
import pyvista as pv
import tifffile as tf

from collections import namedtuple

#%% I/O

def load_segmentation_tif(fname, labels={'seg': 1}, return_vol=False, verbose=False):
    """Load Amira segmentation saved as tif, and get all segmented points."""
    vol = tf.imread(fname) # load data (3d tif, or use mrcfile for mrc)
    vol = np.transpose(vol, axes=[2,1,0]) # Sort axes to xyz
    if verbose:
        print("Input volume shape is {}".format(vol.shape))
    
    # Extract points with labels saved in the dictionary "labels"
    points = {name: np.vstack(np.where(vol == label)).T for name, label in labels.items()}
    
    if return_vol:
        return points, vol
    
    return points
        
    
#%% Simple math

def gcd_floats(a,b):
    """Greatest common divisor of two floats a & b."""
    
    def get_float_decimals(x: float):
        """Get number of decimals of a float."""
        str_rep = str(x).split(".")
        if len(str_rep) == 1:
            return 0        
        return len(str_rep[1])
    
    dec_a = get_float_decimals(a)
    dec_b = get_float_decimals(b)
    f = 10**max(dec_a, dec_b) # factor to multiply the numbers with
    gcd_tmp = np.gcd(int(a*f), int(b*f))
    gcd = gcd_tmp / f
    return gcd


#%% Operations on vectors

def normalize_vector(x, axis=-1):
    """Normalize arrays of vectors along the last axis."""
    return (x.T / np.linalg.norm(x, axis=axis)).T

def angle_between_vectors(v0, v1, normalized=False, degrees=True):
    """
    Calculate the angle between two vectors v0 and v1.

    Parameters
    ----------
    v0 : numpy.ndarray
        First vector, typical shape (2,) or (3,).
    v1 : numpy.ndarray
        Second vector, typical shape (2,) or (3,).
    normalized : bool, optional
        Indicate whether v0 and v1 are both normalized already. The default is False.
    degrees : bool, optional
        Return angle in degrees (alternative rad). The default is True.

    Returns
    -------
    angle : numpy.float64
        Angle between v0 and v1.

    """
    if not normalized:
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
    cos_angle = np.dot(v0,v1)
    angle = np.arccos(cos_angle)
    if degrees:
        return np.rad2deg(angle)
    return angle  

def angle_between_vectors_batch(v0,v1, normalized=False, degrees=True):
    """
    Calculate the angle between two vector arrays v0 and v1.
    
    If v0 and v1 are two 2d arrays of equal shape ((n,3), (n,3)), 
    angles are calculated element-wise and an (n,) array of angles is returned.
    If v0 is an array of vectors (n,3) and v1 one vector (3,), angles are 
    calculated between each vector in v0 and v1.

    Parameters
    ----------
    v0 : numpy.ndarray
        Array of (n,3) or (n,2) vectors.
    v1 : numpy.ndarray
        Array of (m,3) or (m,2) vectors with m = n or m = 1.
    normalized : bool, optional
        Indicate whether v0 and v1 are both normalized already. The default is False.
    degrees : bool, optional
        Return angle in degrees (alternative rad). The default is True.

    Returns
    -------
    angles : numpy.float64
        Array of angles between v0 and v1 vectors.

    """
    if not normalized:
        v0 = normalize_vector(v0)
        v1 = normalize_vector(v1)
    if v0.shape == v1.shape:
        # Einsum allows elementwise dot product
        cos_angles = np.einsum('ij,ij->i', v0, v1)
    elif len(v1.shape)==1:
        cos_angles = np.einsum('ij,j->i', v0, v1)
    else :
        print('Vector sizes do not match.')
        return None
    
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    
    return np.rad2deg(angles) if degrees else angles

def rotation_matrix_from_vectors(v0, v1):
    """
    Find the rotation matrix that aligns v0 to v1.

    Code from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    math explained in: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Parameters
    ----------
    v0 : numpy.ndarray
        A 3d "source" vector.
    v1 : numpy.ndarray
        A 3d "destination" vector.

    Returns
    -------
    rotation_matrix : numpy.ndarray
        A transform matrix (3x3) which when applied to v0, aligns it with v1.

    """
    # Normalize vectors
    a = (v0 / np.linalg.norm(v0)).reshape(3) 
    b = (v1 / np.linalg.norm(v1)).reshape(3)
    # Calculate cross and dot products of normalized vecotrs
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    # Calculate rotation matrix
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

#%% Plotting

def bplot(mesh, scalars=None):
    """Plot a mesh colored by scalars using Pyvista's backgroundplotter."""
    p0 = BackgroundPlotter()
    p0.enable_eye_dome_lighting()
    p0.add_mesh(mesh, scalars=scalars)
    return p0
    
    

#%% Array operations

def sum_over_window(array, win_len, axis=-1):
    """Sum a 2d array in a sliding-window fashion with window length win_len.
    
    Uses scipy.signal.convolve internally.
    """
    if axis ==0:
        summed_array = scipy.signal.convolve(array.T, np.atleast_2d(np.ones(win_len)), mode='valid').T
    else:    
        summed_array = scipy.signal.convolve(array, np.atleast_2d(np.ones(win_len)), mode='valid')
    return summed_array

def replace_nans_with_mean(array):
    """In an array of continuous values, replace NaNs with mean of adjacent values."""
    array_new = array.copy()
    nan_ids = np.where(np.isnan(array))[0]
    for idx in nan_ids:
        if idx == 0:
            array_new[0] = array_new[1]
        elif idx == len(array) -1:
            array_new[idx] = array_new[idx-1]
        else :                
            array_new[idx] = np.nanmean(array[idx-1:idx+2])
    
    return array_new

def find_successive_ones(bool_array, min_count, axis=-1):
    """
    Find the first successive occurence with a given length (min_count) of ones in a 2d array along the given axis.
    
    Can be used e.g. for a histogram mask.

    Parameters
    ----------
    bool_array : np.ndarray
        Array containing ones and zeros, e.g. a mask.
    min_count : int
        Min successive occurence of ones.
    axis : int, optional
        Axis along which to check for successive ons. The default is -1.

    Returns
    -------
    first_ids : list
        List with ids for first occurence of the successive ones. None if no such stretch of ones was detected.

    """
    # Make sliding window sum
    summed_array = sum_over_window(bool_array, win_len=min_count, axis=axis)
    # Find first occurence of min_count along the desired axis
    min_count_array = (summed_array >= min_count)
    first_ids = np.argmax(min_count_array, axis=axis)
    # for rows which are completely zero, put row length (of original array) as first id
    first_ids[np.sum(min_count_array, axis=axis) == 0] = bool_array.shape[axis]
    #first_ids = [None if s==0 else i for i,s in zip(first_ids, np.sum(min_count_array, axis=axis))]

    return first_ids


#%% Operations on point clouds

def PCA(data):
    """
    Perform a Principal Component Analysis on the input array.
    
    Source: https://dev.to/akaame/implementing-simple-pca-using-numpy-3k0a
    
    Parameters
    ----------
    data : numpy.ndarray
        Input array of shape (n,3) or (n,2).

    Returns
    -------
    center: numpy.ndarray
        Mean of coordinates, estimating the center of the point cloud.
    (eigvec, eigval) : tuple of numpy.ndarrays
        Eigen vectors and eigen values, sorted by descending eigen values.
        Eigen vectors are the columns of eigvec, with first eigenvector = eigvec[:,0].
    """
    # Center Data Points
    center =  data.mean(axis=0) # Calculate center
    data = data.copy() - center
    
    # Get Covariance Matrix
    cov = np.cov(data.T) / data.shape[0]
    
    # Perform Eigendecomposition on Covariance Matrix
    eigval, eigvec = np.linalg.eig(cov)
    
    # Sort Eigenvectors According to Eigenvalues
    idx = eigval.argsort()[::-1] # Sort descending and get sorting indices
    eigval = eigval[idx] # Use indices on eigvals
    eigvec = eigvec[:,idx] # Use indices on eigen vectors
    
    return center, (eigvec, eigval)



def get_outlier_points_dbscan(points, dbscan_eps=2, n_main_clusters=1, 
                              min_cluster_size=None):
    """
    Cluster points and return ids of all points that are not in the main cluster(s).
    
    Parameters
    ----------
    points : numpy.ndarray
        DESCRIPTION.
    dbscan_eps : float, optional
        DBSCAN epsilon: maximum distance for two points to be considered to be 
        in each other's neighborhood. The default is 2.
    n_main_clusters : int, optional
        Expected number of main classes. Overruled by min_cluster_size if one 
        is given. The default is 1.
    min_cluster_size : int, optional
        Minimum number of points for a cluster to be kept. If None, the largest 
        n = n_main_clusters clusters are kept. The default is None.

    Returns
    -------
    outlier_ids : numpy.array
        Ids of outlier points. Empty array if no outliers are found.

    """
    # Cluster points using DBSCAN
    cluster_labels = DBSCAN(eps=dbscan_eps).fit_predict(points)
    # Get unique labels 
    unique_labels, unique_counts = np.unique(cluster_labels, return_counts=True)
    # If there's just one label, return empty array
    if len(unique_labels) == 1:
        outlier_ids = np.array([])
    
    # If a minimum cluster size is given, return point ids of all clusters smaller than the given size
    elif min_cluster_size is not None:
        labels_discard = unique_labels[unique_counts < min_cluster_size]
        outlier_ids = np.where(cluster_labels == labels_discard)[0]
        
    # Else, give ids of points not in largest cluster(s)
    else:
        # Find label of main cluster(s)
        label_main_clusters = unique_labels[np.argsort(unique_counts)[-n_main_clusters:]]
        outlier_ids = np.where(~np.isin(cluster_labels, label_main_clusters))[0]
    
    return outlier_ids

def clean_pointlist_clustering(point_list, min_cluster_size=500, dbscan_eps=5):
    """Identify outliers in lists of point arrays.
    
    Point arrays in the list are treated as matched, if one point is identified as outlier,
    the whole pair or group is treated as an outlier. Used e.g. to further clean refined
    inner / outer points of rough double membrane segmentations.
    """
    out_ids_list = []
    for points in point_list:
        out_ids_tmp =  get_outlier_points_dbscan(points, min_cluster_size=min_cluster_size, dbscan_eps=dbscan_eps) 
        out_ids_list.append(out_ids_tmp)
        
    ids_all = np.concatenate(tuple(out_ids_list), axis=0)
    ids_unique = np.unique(ids_all)
    
    return ids_unique

def points_in_hull(hull, query_points, tolerance=1e-12, use_subset=False):
    """
    Check if points are within a convex hull.
    
    Adapted from: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/42165596#42165596
    Convex hull can be generated with scipy.spatial.ConvexHull.

    Parameters
    ----------
    hull : qhull (generate with scipy.spatial.ConvexHull)
        Convex Hull for testing query points.
    query_points : numpy.ndarray
        (N,3) array of query point coordinates.
    tolerance : float, optional
        Tolerance for checking if point belongs to hull. The default is 1e-12.
    use_subset : bool, optional
        If True, exclude points first which are completely out of the range of the convex hull. The default is False.

    Returns
    -------
    in_hull : numpy.ndarray
        (N,1) boolean array showing for each point whether it is in the hull.
    query_points : numpy.ndarray, optional
        If use_subset=True, these are the actual points checked.
    sub_ids : numpy.ndarray, optional
        If use_subset=True, sub_ids are the ids of the original points checked against the hull.

    """
    if len(query_points) == 0:
        return []
    if use_subset:
        points_orig = query_points.copy()
        sub_ids = np.where(np.all(np.logical_and(points_orig <= np.max(hull.points, axis=0), 
                                                 points_orig >= np.min(hull.points, axis=0)), axis=1))[0]
        query_points = points_orig[sub_ids]
    # Check if points are in the convex hull using hull equations    
    in_hull = np.all(np.add(np.dot(query_points, hull.equations[:,:-1].T),hull.equations[:,-1]) <= tolerance, axis=1)
    if use_subset:
        return in_hull, query_points, sub_ids
    
    return in_hull
    
def clean_points_opening(points, vol_shape, structure=np.ones((3,3,3)), iterations=1, plot_results=False,
                        return_volume=False, return_points=True):
    """
    Clean a point cloud by binary opening.

    Parameters
    ----------
    points : numpy.ndarray
        (N,3) array of points.
    vol_shape : tuple
        (3,) tuple indicating the shape of the full volume.
    structure : numpy.ndarray, optional
        Structure element used for binary opening. The default is np.ones((3,3,3)) (a 3x3 cube).
    iterations : int, optional
        Number of iterations of binary opening. The default is 1.
    plot_results : bool, optional
        Indicate whether resulting volume should be plotted using pyvista.Plotter. The default is False.
    return_volume : bool, optional
        If true, return volume. The default is False.
    return_points : TYPE, optional
        If true, return point coordinates after opening. The default is True.

    Returns
    -------
    points_res : numpy.ndarray, if return_points = True
        Coordinates of points after binary opening.
    vol_1 : numpy.ndarray, if return_volume = True
        Volume of shape vol_shape with 0 = background and 1 = cleaned points.
    """
    # Make a volume of zeros and put in the original points
    vol_0 = np.zeros(vol_shape)
    vol_0[points[:,0], points[:,1], points[:,2]] = 1
    # Do the cleaning
    vol_1 = scipy.ndimage.binary_opening(vol_0, structure=structure, iterations=iterations)
    # Return volume if wanted
    if return_volume and not return_points:
        return vol_1
    # Get the points
    points_res = np.vstack(np.where(vol_1 == 1)).T
    if plot_results:
        p0 = pv.Plotter()
        p0.enable_eye_dome_lighting()
        #p0.add_mesh(points, color='grey', opacity=0.5)
        p0.add_mesh(points_res, color='white')
        p0.show(window_size=[400,300])
    
    if return_volume:
        return points_res, vol_1
    return points_res


    
#%% Data type related operations

def namedtuple_to_dict(tup):
    """Turn a named tuple into a dictionary, e.g. for saving. Also works for nested structures."""
    if isinstance(tup, (tuple)) and hasattr(tup, '_fields'):
        out = dict(tup._asdict())
    else:
        out = tup
    for k,v in out.items():
        if (isinstance(v, (tuple)) and hasattr(v, '_fields')) or isinstance(v, dict):
            out[k] = namedtuple_to_dict(v)
    return out    

# Functions for comparing dictionaries
def dicts_keys_equal(d1, d2):
    """Return True if all keys are the same."""
    return all(k in d2 for k in d1) and all(k in d1 for k in d2)

def dicts_values_equal(d1, d2):
    """Return True if all values are the same."""    
    out = []
    for k in d1:
        if type(d1[k]) == dict and type(d2[k]) == dict:
            if not dicts_keys_equal(d1[k], d2[k]):
                out.append(False)
            else:
                out.append(dicts_values_equal(d1[k], d2[k]))
        else:
            if type(d1[k]) in (np.ndarray, pv.core.pyvista_ndarray):
                out.append(np.array_equal(d1[k], d2[k]))
            else:
                test = d1[k] == d2[k]
                if type(test) in (bool, np.bool_):
                    out.append(test)
                else:
                    out.append(all(test))            
    return out
        
def dicts_all_equal(d1, d2):
    """Return True if all keys and values are the same."""
    if not dicts_keys_equal(d1, d2):
        return False
    return all(dicts_values_equal(d1, d2))    
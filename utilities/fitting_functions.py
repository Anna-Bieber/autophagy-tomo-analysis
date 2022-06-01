#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit different geometric structures into point clouds, and create representations of these structures.

Uses pyvista to create meshes, and least squares algorithms 
found at http://www.juddzone.com/ALGORITHMS/ @author: Tom Judd

@author Anna Bieber
"""
# Imports
import numpy as np
from numpy.linalg import eig, inv
import pyvista as pv
import math
from collections import namedtuple
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev, bisplrep, bisplev
from scipy.optimize import minimize
from functools import partial

from tqdm import tqdm
import itertools


from utility_functions import normalize_vector, PCA, rotation_matrix_from_vectors
from mesh_functions import make_2d_grid_faces


#%% Line / plane fit

def fit_straight_line(points):
    """Fit a straight line into a set of 3D points using PCA."""
    center, (eigvec, eigval) = PCA(points)
    line_dir = eigvec[:,0] / np.linalg.norm(eigvec[:,0])
    return center, line_dir

def fit_plane(points, reference_point, N_to_ref=True):
    """
    Fit a plane into points with the normal pointing into the direction of the reference point.
    
    If N_to_ref=False, normal will point away from reference point. 
    If return_mesh, also returns a pyvista mesh of the plane. The center and normal of the plane are saved as 
    named tuples and can be called e.g. by plane.center which is equivalent to plane[0].
    """
    center, (evecs_0, evals_0) = PCA(points)
    normal = normalize_vector( evecs_0[:,-1] )
    
    plane_tuple = namedtuple('plane', ['center', 'normal'])
    plane = plane_tuple(center, normal)
    
    # Normal direction: should point towards reference point
    plane_to_ref = reference_point - center
    if N_to_ref and np.dot(plane_to_ref, normal) < 0:
        normal *= -1
    elif np.dot(plane_to_ref, normal) > 0 and not N_to_ref:
        normal *= -1

    return plane   

def dist_to_plane_signed(points, plane):
    """Distances of points to a plane defined by a named tuple.
    
    Positive distances indicate points lying in direction of the normals.
    """
    return np.dot(points-plane.center, plane.normal)

#%% Surface fits (polynomial or spline)

# Polynomial surface fit
# Source: https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent, 17.12.2020

# Function for fitting a polynomial surface in 3D
def polyfit2d(points, order=3):
    """
    Fit a polynomial surface of given order into points.
    
    The resulting parameters stored in the array m can be used in polyval2d to 
    generate points in the polynomial surface.
    Source: https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent, 
    visited 17.12.2020.
    """
    x,y,z = points.T
    # Prepare value matrix
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    """Calculate z values for x,y coordinates and polynomial surface parameters determined e.g. with polyfit2d."""
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def transformation_params_xy_plane(points, align_surf_z=False):
    """Given points lying roughly in a plane, determine a new coordinate system in which the plane corresponds to xy."""
    # PCA of points -> transform into coordinate system of eigen vectors
    center, (eigvec, eigval) = PCA(points)  # PCA, eigvecs should already be orthonormal
    # If needed, align tranformation vectors so one of the first two is orthogonal to z first
    if align_surf_z:
        # Find out if eigvec 0 or 1 is more orthogonal to z
        rotvec_id = np.argmin( np.dot(np.array([0,0,1]), eigvec)[:2] )
        rotvec_orig = eigvec[:, rotvec_id]
        # Target vector: get rid of z component of vector & normalize
        rotvec_target = normalize_vector(rotvec_orig - np.dot(rotvec_orig, np.array([0,0,1])))
        # Calculate rotation matrix
        rotmat_vecs = rotation_matrix_from_vectors(rotvec_orig, rotvec_target)
        # Apply rotation matrix to eigen vectors
        tf_vectors = rotmat_vecs.dot(eigvec)
    else:
        tf_vectors = eigvec # Eigvecs are the x,y,z axes of the new coordinate system
    
    return center, tf_vectors # Columns of tf_vectors are x,y,z
    

def polynomial_surface_fit_v1(points, order=2, output_step=1, output_extrapol=None, align_surf_z=False,
                           return_grid_info=False):
    """
    Fit a polynomial surface into points arranged in a plane-like geometry.

    Parameters
    ----------
    points : numpy.ndarray
        Points to perform the fit on.
    order : int, optional
        Polynomial order. The default is 2.
    output_step : float, optional
        xy step size for output points. The default is 1.
    output_extrapol : float or None, optional
        If given, extrapolate the output surface by this fraction of its original size. The default is None.
    align_surf_z : bool, optional
        Whether one of the xy axes of the output plane should be aligned to the global z axis. The default is False.
    return_grid_info : bool, optional
        If true, extra info on the output grid is returned, e.g. its id array and edge ids. The default is False.

    Returns
    -------
    fit_surf : pyvista.PolyData
        Mesh of the output polynomial surface.
    grid_info : namedtuple, optional (if return_grid_info = True)
        Extra info on the output mesh.
    """    
    # Get transformation parameters so point cloud lies in xy plane around (0,0,0)
    center, tf_vectors = transformation_params_xy_plane(points, align_surf_z=align_surf_z)
    
    # Apply transformation to the points. Columns of tf_vectors are the x,y,z axes of the new coordinate system
    points_shift = points - center
    points_tf = np.dot(points_shift, tf_vectors) 
    
    # Polynomial fit of transformed points
    m = polyfit2d(points_tf, order=order)
    
    # Construct xy grid
    xmin_tf = np.floor(points_tf[:,0].min())
    xmax_tf = np.ceil(points_tf[:,0].max())
    ymin_tf = np.floor(points_tf[:,1].min())
    ymax_tf = np.ceil(points_tf[:,1].max())
    
    # Enlarge range by output_extrapol in all directions if wished
    if output_extrapol is not None:
        xlen = xmax_tf - xmin_tf
        xmin_tf -= np.round(output_extrapol*xlen)
        xmax_tf += np.round(output_extrapol*xlen)
        ylen = ymax_tf - ymin_tf
        ymin_tf -= np.round(output_extrapol*ylen)
        ymax_tf += np.round(output_extrapol*ylen)
    
    xx, yy = np.meshgrid(np.arange(xmin_tf, xmax_tf+1, step=output_step),
                         np.arange(ymin_tf, ymax_tf+1, step=output_step))
    
    # Calculate z values and put points together           
    zz = polyval2d(xx, yy, m)
    fit_points_tf = np.c_[xx.flatten(),yy.flatten(), zz.flatten()]

    # Transform fit points back into original coordinate system
    fit_points = center  + (np.matmul(np.atleast_2d(fit_points_tf[:,0]).T, np.atleast_2d(tf_vectors[:,0]))
                            + np.matmul(np.atleast_2d(fit_points_tf[:,1]).T, np.atleast_2d(tf_vectors[:,1]))
                            + np.matmul(np.atleast_2d(fit_points_tf[:,2]).T, np.atleast_2d(tf_vectors[:,2])) )
    
    # Make a pyvista surface from the transformed fit points and calculate its normals
    # New: construct faces from the intrinsic order of the grid, rather than using delaunay triangulation
    faces = make_2d_grid_faces(xx.shape)
    fit_surf = pv.PolyData(fit_points, faces)
    fit_surf.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    
    # If needed, return info on the rectangular grid that contains the indices
    if return_grid_info:
        grid_info_tuple = namedtuple('rectangular_grid', ['id_array', 'edge_ids', 'edge_points'])
        n_points = fit_points.shape[0]
        grid_shape = xx.shape
        grid_id_array = np.arange(n_points).reshape(grid_shape)
        edge_ids = np.array([grid_id_array[0,0], grid_id_array[0,-1], grid_id_array[-1,0], grid_id_array[-1,-1]]).astype(int)
        edge_points = fit_points[edge_ids]
        grid_info = grid_info_tuple(grid_id_array, edge_ids, edge_points)
        
        return fit_surf, grid_info
    
    return fit_surf

#%% Spline fitting: curves and surfaces

def fit_curve_spline(points, smooth_factor=500):
    """Fit a spline into points on a curve."""
    tck, u = splprep([points[:,0], points[:,1], points[:,2]], u=points[:,2], s=smooth_factor)
    spline_points = splev(u, tck)
    spline_point_array = np.r_[spline_points].T
    return spline_point_array

def fit_curve_spline_oversampled(points, smooth_factor=500, oversample_factor=5):
    """Fit a spline into points on a curve, oversampling by the given factor."""
    tck, u = splprep([points[:,0], points[:,1], points[:,2]], s=smooth_factor)
    spline_points = splev(np.linspace(0,1,int(len(u)*oversample_factor)), tck) # generate oversampled spline points
    spline_point_array = np.r_[spline_points].T
    return spline_point_array

def fit_surface_spline(points, output_step=1, output_extrapol=None, align_surf_z=False,
                       return_grid_info=False):
    """
    Fit a spline surface into points arranged in a plane-like geometry.

    Parameters
    ----------
    points : numpy.ndarray
        Points to perform the fit on.
    output_step : float, optional
        xy step size for output points. The default is 1.
    output_extrapol : float or None, optional
        If given, extrapolate the output surface by this fraction of its original size. The default is None.
    align_surf_z : bool, optional
        Whether one of the xy axes of the output plane should be aligned to the global z axis. The default is False.
    return_grid_info : bool, optional
        If true, extra info on the output grid is returned, e.g. its id array and edge ids. The default is False.

    Returns
    -------
    fit_surf : pyvista.PolyData
        Mesh of the output polynomial surface.
    grid_info : namedtuple, optional (if return_grid_info = True)
        Extra info on the output mesh.
    """
    # Get transformation parameters so point cloud lies in xy plane around (0,0,0)
    center, tf_vectors = transformation_params_xy_plane(points, align_surf_z=align_surf_z)
    
    # Apply transformation. Columns of tf_vectors are the x,y,z axes of the new coordinate system
    points_shift = points - center
    points_tf = np.dot(points_shift, tf_vectors) 
    
    # Spline fit of transformed points
    tck = bisplrep(points_tf[:,0], points_tf[:,1], points_tf[:,2])
    
    # Construct xy grid
    xmin_tf = np.floor(points_tf[:,0].min())
    xmax_tf = np.ceil(points_tf[:,0].max())
    ymin_tf = np.floor(points_tf[:,1].min())
    ymax_tf = np.ceil(points_tf[:,1].max())
    
    # Enlarge range by output_extrapol in all directions if wished
    if output_extrapol is not None:
        xlen = xmax_tf - xmin_tf
        xmin_tf -= np.round(output_extrapol*xlen)
        xmax_tf += np.round(output_extrapol*xlen)
        ylen = ymax_tf - ymin_tf
        ymin_tf -= np.round(output_extrapol*ylen)
        ymax_tf += np.round(output_extrapol*ylen)
    
    xx, yy = np.mgrid[xmin_tf : xmax_tf+1 : output_step,
                      ymin_tf : ymax_tf+1 : output_step]

    # Calculate z values and put points together           
    zz = bisplev(xx[:,0], yy[0,:], tck)
    fit_points_tf = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    # Transform fit points back into original coordinate system
    fit_points = center  + (np.matmul(np.atleast_2d(fit_points_tf[:,0]).T, np.atleast_2d(tf_vectors[:,0]))
                            + np.matmul(np.atleast_2d(fit_points_tf[:,1]).T, np.atleast_2d(tf_vectors[:,1]))
                            + np.matmul(np.atleast_2d(fit_points_tf[:,2]).T, np.atleast_2d(tf_vectors[:,2])) )
    
    # Make a pyvista surface from the transformed fit points and calculate its normals
    faces = make_2d_grid_faces(xx.shape)
    fit_surf = pv.PolyData(fit_points, faces)
    # fit_cloud = pv.PolyData(fit_points)
    # fit_surf = fit_cloud.delaunay_2d()
    fit_surf.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    
    # If needed, return info on the rectangular grid that contains the indices
    if return_grid_info:
        grid_info_tuple = namedtuple('rectangular_grid', ['id_array', 'edge_ids', 'edge_points'])
        n_points = fit_points.shape[0]
        grid_shape = xx.shape
        grid_id_array = np.arange(n_points).reshape(grid_shape)
        edge_ids = np.array([grid_id_array[0,0], grid_id_array[0,-1], grid_id_array[-1,0], grid_id_array[-1,-1]]).astype(int)
        edge_points = fit_points[edge_ids]
        grid_info = grid_info_tuple(grid_id_array, edge_ids, edge_points)
        
        return fit_surf, grid_info
    
    return fit_surf

#%% Sphere fits (least squares)

def ls_sphere(xx,yy,zz):
    """
    Least squares sphere fit version 1.
    
    Found at http://www.juddzone.com/ALGORITHMS/least_squares_sphere.html .
    """
    asize = np.size(xx)
    #print('Sphere input size is ' + str(asize))
    J=np.zeros((asize,4))
    ABC=np.zeros(asize)
    K=np.zeros(asize)
 
    for ix in range(asize):
        x=xx[ix]
        y=yy[ix]
        z=zz[ix]
     
        J[ix,0]=x*x + y*y + z*z
        J[ix,1]=x
        J[ix,2]=y
        J[ix,3]=z
        K[ix]=1.0
 
    K=K.transpose() #not required here
    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=np.linalg.inv(JTJ)
 
    ABC= np.dot(InvJTJ, np.dot(JT,K))
    #If A is negative, R will be negative
    A=ABC[0]
    B=ABC[1]
    C=ABC[2]
    D=ABC[3]
 
    xofs=-B/(2*A)
    yofs=-C/(2*A)
    zofs=-D/(2*A)
    R=np.sqrt(4*A + B*B + C*C + D*D)/(2*A)
    if R < 0.0: R = -R
    return (xofs,yofs,zofs,R)
  
def ls_sphere_2(spX,spY,spZ):
    """
    Least squares sphere fit version 2.
    
    Found at https://jekel.me/2015/Least-Squares-Sphere-Fit/
    
    Approach:
    Starting point: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
    Rearranged to: x^2 + y^2 + z^2 = 2x*x0 + 2y*y0 + 2z*z0 + 1*(r^2 -x0^2 -y0^2 -z0^2)
    --> Matrix:           f        =  A c
    with  f: nx1 vector with (xi^2 + yi^2 + zi^2) in i=1:n rows
          A: nx4 matrix with rows (2xi, 2yi, 2zi, 1)
          c: 4x1 target matrix [x0, y0, z0, (r^2 - x0^2 - y0^2 - z0^2)]

    """        
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residuals, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2], residuals

def sphere_fit_rmse(points, center, radius):
    """Calculate root mean square error of a sphere fit to points."""
    distances_to_center = np.sqrt( np.sum((points - center)**2, axis=1) )
    residuals = distances_to_center - radius
    residual_sum_squares = np.sum( np.power(residuals, 2) )
    rmse = np.sqrt(residual_sum_squares / len(residuals))
    return rmse

def ls_sphere_3(points):
    """
    Least squares fit of sphere to points.
    
    Modified code following approach described in:
    https://jekel.me/2015/Least-Squares-Sphere-Fit/
    
    Approach:
    Starting point: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
    Rearranged to: x^2 + y^2 + z^2 = 2x*x0 + 2y*y0 + 2z*z0 + 1*(r^2 -x0^2 -y0^2 -z0^2)
    --> Matrix:           f        =  A c
    with  f: nx1 vector with (xi^2 + yi^2 + zi^2) in i=1:n rows
          A: nx4 matrix with rows (2xi, 2yi, 2zi, 1)
          c: 4x1 target matrix [x0, y0, z0, (r^2 - x0^2 - y0^2 - z0^2)]

    
    Parameters
    ----------
    points : numpy.ndarray
        (n,3) array of point xyz coordinates.

    Returns
    -------
    center : numpy.ndarray
        xyz coordinates of sphere center.
    radius : float
        Sphere radius.
    rmse : float
        Root mean square error of fit.

    """
    # Assemble the A matrix
    A = np.append(2*points, np.ones( (points.shape[0],1) ), axis=1 ) # (2xi, 2yi, 2zi, 1)
    # Assemble f
    f = np.sum( np.power(points,2), axis=1) # x^2 + y^2 + z^2
    
    # Solve linear equation system Ac = f using least-squares algorithm
    c, residuals, rank, singval = np.linalg.lstsq(A,f)
    # Extract the center (first three values in c)
    center = c[0:3]
    # solve for the radius, c[3] = r^2 - x0^2 - y0^2 - z0^2
        # --> radius = sqrt( c[3] + x0^2 + y0^2 + z0^2 )
    radius = math.sqrt( c[3] + np.sum( np.power(center,2) ) )
    
    # determine RMSE by comparing distances of points to center to the radius
    rmse = sphere_fit_rmse(points, center, radius)
    
    return center, radius, rmse
    
#%% Least squares fit to a 3D-ellipsoid    

# http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
# For better results, use ellipsoid fitting functions defined in fit_ellipsoid_weiss.py

def ls_ellipsoid(xx,yy,zz):
    """
    Least-squares ellipsoid fit, part 1.
    
    Taken from http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html.
    """
    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:,np.newaxis]
    y = yy[:,np.newaxis]
    z = zz[:,np.newaxis]
 
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones
 
    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.
 
    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=np.linalg.inv(JTJ);
    ABC= np.dot(InvJTJ, np.dot(JT,K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa=np.append(ABC,-1)
 
    return (eansa)

def polyToParams3D(vec, printMe=False):
    """Least-squares ellipsoid fit, part 2.
    
    Taken from http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html.
    
    convert the polynomial form of the 3D-ellipsoid to parameters
    center, axes, and transformation matrix
    vec is the vector whose elements are the polynomial
    coefficients A..J
    returns (center, axes, rotation matrix)

    Algebraic form: X.T * Amat * X --> polynomial form
    """
    Amat=np.array(
    [
    [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
    [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
    [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
    [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
    ])
    
    if printMe: print('\nAlgebraic form of polynomial\n',Amat)
    
    #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3=Amat[0:3,0:3]
    A3inv=inv(A3)
    ofs=vec[6:9]/2.0
    center=-np.dot(A3inv,ofs)
    if printMe: print('\nCenter at:',center)
    
    # Center the ellipsoid at the origin
    Tofs=np.eye(4)
    Tofs[3,0:3]=center
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))
    if printMe: print('\nAlgebraic form translated to center\n',R,'\n')
    
    R3=R[0:3,0:3]
    R3test=R3/R3[0,0]
    if printMe: print('normed \n',R3test)
    s1=-R[3, 3]
    R3S=R3/s1
    (el,ec)=eig(R3S)
    
    recip=1.0/np.abs(el)
    axes=np.sqrt(recip)
    if printMe: print('\nAxes are\n',axes  ,'\n')
    
    inve=inv(ec) #inverse is actually the transpose here
    if printMe: print('\nRotation matrix\n',inve)
    return (center,axes,inve)

#%% Generate sphere and ellipsoid points and meshes

def points_to_surface(points):
    """Turn point array into pyvista mesh.
    
    Code found in https://stackoverflow.com/questions/47485235/i-want-to-make-evenly-distributed-sphere-in-vtk-python
    """
    # Make points into Pyvista PolyData
    point_cloud = pv.PolyData(points)
    # Delaunay triangulation and surface extraction
    surf = point_cloud.delaunay_3d().extract_surface()
    
    return surf

def generate_ellipsoid(radii, center, rotation_matrix, n_angles=60, 
                       return_mesh=False):
    """
    Generate ellipsoid points with evenly spaced angles.
    
    This results in a higher point density close to the poles. For more evenly distributed points,
    use of generate_ellipsoid_even is recommended.
    """
    # generate angles
    u = np.linspace(0.0, 2.0 * np.pi, n_angles)
    v = np.linspace(0.0, np.pi, n_angles)
    # points on xyz ellipsoid around center
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # rotate and shift points
    for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation_matrix) + center
    
    # flatten matrices and stack them to xyz
    ellipsoid_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    if return_mesh:
        return points_to_surface(ellipsoid_points)
    
    return ellipsoid_points


def generate_ellipsoid_even(radii, center, rotation_matrix, n_points=1000, 
                            return_mesh=False, return_angles=False):
    """
    Generate ellipsoid points or mesh from ellipsoid parameters.
    
    Surface generation found in: https://stackoverflow.com/questions/47485235/i-want-to-make-evenly-distributed-sphere-in-vtk-python
    Initial point generation is done with a Fibonacci lattice / golden spiral 
    approach to get points that are (relatively) evenly distributed on a unit 
    sphere.
    Coordinates are then stretched by the radii, rotated by the rotation matrix 
    and shifted by the center.
    If return_mesh=True, a surface mesh is finally generated from the points 
    using delaunay triangulation implemented in pyvista.


    Parameters
    ----------
    radii : numpy.ndarray
        (3,) array of ellipsoid radii / axis lengths.
    center : numpy.ndarray
        Coordinates of ellipsoid center (3,).
    rotation_matrix : numpy.ndarray
        Ellpsoid rotation matrix.
    n_points : int, optional
        Number of points in final structure. The default is 1000.
    return_mesh : bool, optional
        If True, returns a pyvista mesh generated from the points. The default is False.

    Returns
    -------
    points : numpy.ndarray, if return_mesh = False (default)
        A (n_points,3) array of ellipsoid points.
        If return_mesh = True, a pyvista mesh of the ellipsoid surface is returned
        instead. The ellipsoid points are then accessible through mesh.points.

    """
    # Generate coordinates distributed by a Fibonacci pattern on the base ellipsoid.    
    indices = np.arange(0, n_points, dtype=float) + 0.5
    
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    theta = theta % (2*np.pi) # put theta into range (0, 2*pi)
    
    x =  np.cos(theta) * np.sin(phi) * radii[0]
    y = np.sin(theta) * np.sin(phi) * radii[1]
    z = np.cos(phi) * radii[2]
    
    # combine points
    points = np.c_[x,y,z]
    # rotate points with rotation matrix
    points = np.dot(points, rotation_matrix)
    # Shift points by center
    points += center
    
    if return_angles and not return_mesh:
        return points, theta, phi
    elif return_angles and return_mesh:
        return points_to_surface(points), theta, phi
    
    if return_mesh:
        return points_to_surface(points)
    
    return points

def generate_sphere(radius, center, n_angles=60):
    """
    Generate sphere points with evenly spaced angles.
    
    This results in higher point density close to the poles. For more homogeneous point distributions,
    use generate_sphere_even.
    """
    # generate angles
    u = np.linspace(0.0, 2.0 * np.pi, n_angles)
    v = np.linspace(0.0, np.pi, n_angles)
    # calculate points
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    # flatten matrices and stack them to xyz
    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    return sphere_points

def generate_sphere_even(radius, center, n_points=1000, return_mesh=False):
    """
    Generate points distributed relatively evenly on a sphere.
    
    The points follow a Fibonacci / golden spiral.
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5
    
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x =  np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius
    
    # combine points
    points = np.c_[x,y,z]
    # Shift points by center
    points += center
    
    if return_mesh:
        return points_to_surface(points)
    
    return points
    


def mean_dist_to_fit(orig_points, fit_points):
    """Calculate the mean distance of points to their corresponding fit points."""
    # Make KDTree of fit points
    tree = cKDTree(fit_points)
    # Query minimum distance of orig points to fit
    distances, closest_point_idx = tree.query(orig_points)
    
    distance_mean = np.mean(distances)
    
    return distance_mean

#%% Ellipsoid characterization

def ellipsoid_area_thomsen(axes):
    """
    Approximation of ellipsoid surface area after Knud Thomsen (2004).

    Maximum error is +-1.061%    
    Source: www.numericana.com/answer/ellipsoid.htm 
    cited e.g. in https://doi.org/10.1016/j.mri.2008.07.018
    """   
    a, b, c = axes
    p = 1.6075 # constant defined by K. Thomsen
    
    area = 4 * np.pi* (( (a*b)**p + (a*c)**p + (b*c)**p ) / 3)**(1/p)    
    
    return area

def ellipsoid_volume(axes):
    """Volume of an ellipsoid."""
    V = 4/3 * np.pi * axes[0] * axes[1] * axes[2] # Volume is 4/3*pi*abc
    return V

def sphericity_of_ellipsoid(axes, verbose=False):
    """
    Calculate the 'classical' sphericity of an ellipsoid.
    
    Sphericity is defined as surface area of sphere with same volume 
    divided by actual surface area.
    """
    # Surface area of ellipsoid
    A_ellipsoid_kt = ellipsoid_area_thomsen(axes) # Determine ellipsoid surface area with Knud Thomsen approximation
    if verbose:
        print("Estimated ellipsoid surface area: {}".format(A_ellipsoid_kt))
    
    # Volume of ellipsoid
    V_ellipsoid = ellipsoid_volume(axes)
    
    # Surface area of corresponding sphere
    r_csphere = ( (3*V_ellipsoid ) / (4 * np.pi) )**(1/3) # radius of corresponding sphere
    A_csphere =  4 * np.pi * r_csphere**2 # Area of corresponding sphere, 4 pi r^2
    if verbose:
        print("Surface area of sphere with same volume: {}".format(A_csphere))
    # Sphericity    
    sphericity = A_csphere / A_ellipsoid_kt
    if verbose:
        print("The sphericity of the best-fitting ellipsoid is {}".format(sphericity))
    
    return sphericity

  
def sphericity_index_ellipsoid(axes):
    """Sphericity index of an ellipsoid as defined in Cruz-Matias et al. 2019, doi: 10.1016/j.jocs.2018.11.005.""" 
    axis_c, axis_b, axis_a = np.sort(axes) 
    sphericity_index = (axis_c**2 / (axis_a*axis_b))**(1/3)
    
    return sphericity_index


def ellipsoid_point_from_angles(theta, phi, center, axes, rotmat):
    """
    Generate points on an ellipsoid from the given angles.
    
    Angle ranges: theta 0->2pi, phi 0->pi
    """
    # Generate point from angles
    x =  np.cos(theta) * np.sin(phi) * axes[0]
    y = np.sin(theta) * np.sin(phi) * axes[1]
    z = np.cos(phi) * axes[2]    
    point = np.array([x,y,z])
    # rotate point with rotation matrix and shift it by the center
    point = np.dot(point, rotmat) + center
    
    return point

#%% Ellipsoid distances and fit rmse

def distance_to_ellipsoid(angles, query_point, center, axes, rotmat):
    """
    Calculate the distance of a query point to a point on an ellipsoid.
    
    The ellipsoid point is defined by angles and ellipsoid parameters.
    angles: tuple of (theta, phi) with theta in [0,2pi] and phi in [0,pi].
    """  
    # Unpack angles
    theta, phi = angles
    # Generate ellipsoid point from angles
    x =  np.cos(theta) * np.sin(phi) * axes[0]
    y = np.sin(theta) * np.sin(phi) * axes[1]
    z = np.cos(phi) * axes[2]    
    point = np.array([x,y,z])
    # rotate point with rotation matrix and shift it by the center
    point = np.dot(point, rotmat) + center
    
    # Calculate distance to query point
    distance = np.linalg.norm(query_point - point)
    
    return distance

def minimum_distance_ellipsoid(query_point, center, axes, rotmat, init_angles=(0,0)):
    """
    Find the closest distance and point on an ellipsoid for a query point.
    
    The closest distance and point are found by minimizing the distance function using scipy.optimize.minimize.
    
    Parameters
    ----------
    query_point : numpy.ndarray
        Coordinates of query point.
    center : numpy.ndarray
        Ellipsoid center.
    axes : numpy.ndarray
        Ellipsoid axes.
    rotmat : numpy.ndarray
        Ellipsoid rotation matrix.
    init_angles : tuple, optional
        Angles describing initial guess for closest ellipsoid point to ellipsoid point. The default is (0,0).

    Returns
    -------
    distance : float
        Distance between query point and the determined closest ellipsoid point.
    closest_ellipsoid_point : numpy.ndarray
        Coordinates of closest ellipsoid point.
    res : scipy.optimize.OptimizeResult
        Complete results of scipy.optimize.minimize. Important attributes are:
            res.x : (theta, phi) angles describing the closest ellipsoid point.
            res.fun = distance (closest distance)
            res.success : Boolean flag indicating whether optimization was successful.
        For a complete description see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    # Minimize the distance over the pair of angles that define the ellipsoid point
    minimizing_function = partial(distance_to_ellipsoid, query_point=query_point, center=center, axes=axes, rotmat=rotmat) 
    res = minimize(minimizing_function, init_angles, bounds=((0, 2*np.pi), (0, np.pi)), tol=1e-6)
    # The distance is the minimized function value
    distance = res.fun    
    # Find the closest ellipsoid point using the angles
    closest_ellipsoid_point = ellipsoid_point_from_angles(*res.x, center, axes, rotmat)
    
    return distance, closest_ellipsoid_point, res    




def ellipsoid_fit_rmse(query_points, axes, center, rotmat, return_full=True):
    """
    Calculate the root mean square error for an ellipsoid fit.
    
    For each of the query poins, determine the closest distance and corresponding 
    point on the ellipsoid by minimizing the distance function.
    
    Parameters
    ----------
    query_points : numpy.ndarray
        Points to check against the ellipsoid fit.
    axes : numpy.ndarray
        Ellipsoid axes lengths.
    center : numpy.ndarray
        Ellipsoid center.
    rotmat : numpy.ndarray
        Ellipsoid rotation matrix.
    return_full : bool, optional
        Whether to return all distances and nearest ellipsoid points. The default is True.

    Returns
    -------
    ellipsoid_rmse : float
        Root mean square error of ellipsoid fit.
    dist_pe : numpy.ndarray, optional, if return_full=True
        Distances of all query points to ellipsoid.
    ell_points_nearest : numpy.ndarray, optional, if return_full=True
        Nearest ellipsoid points for all query points.
    """    
    # Find starting angles for all query points
    ellipsoid_points, ell_theta, ell_phi = generate_ellipsoid_even(axes, center, rotmat, 
                                                                   n_points=2000, return_angles=True)
    ellipsoid_angles = np.c_[ell_theta, ell_phi] # Combine ellipsoid angles into array
    
    # Find closest ellipsoid point for each query point and get the corresponding set of angles
    tree_ell = cKDTree(ellipsoid_points) # Make a KDTree
    _, idx_ell0 = tree_ell.query(query_points) # Find the closest points    
    init_angle_array = ellipsoid_angles[idx_ell0] # Find the corresponding angles
    
    # Prepare output arrays
    dist_pe = np.zeros(query_points.shape[0])
    ell_points_nearest = np.zeros(query_points.shape)
    
    # Run minimization
    for i, point in enumerate(tqdm(query_points)):
        #if i > 10000: break
        init_angles = tuple( init_angle_array[i] )
        dist_pe[i], ell_points_nearest[i,:], res = minimum_distance_ellipsoid(point, center, 
                                                                              axes, rotmat, 
                                                                              init_angles=init_angles)
    
    # Calculate the rmse
    residual_sum_squares = np.sum( np.power(dist_pe, 2) )
    ellipsoid_rmse = np.sqrt(residual_sum_squares / len(dist_pe))
    
    if return_full:
        return ellipsoid_rmse, dist_pe, ell_points_nearest

    return ellipsoid_rmse
    
    
#%% 3D Circle fit and functions needed for it

#!!! Changed input: originally had input (plane_origin, plane_normal), but origin was never used.
def plane_axes(plane_normal):
    """Generate two orthogonal normalized axes in a 3D plane."""
    # normalize normal just to be sure
    plane_normal = plane_normal / np.linalg.norm(plane_normal) 
    # If plane normal is along global z axis, take x and y normal vectors 
    # as plane axes. This case probably doesn't happen with phagophore rim data.
    if plane_normal[0] == 0 and plane_normal[1] == 0:
        plane_axis_0 = np.array([1,0,0])
        plane_axis_1 = np.array([0,1,0])
    # This is the more common case
    else:
        # First axis: z=0, perpendicular to normal
        # with n1x1 + n2x2 = 0 and sqrt(x1^2 + x2^2) = 1, it follows that:
        # x1 = n2*(n1^2+n2^2)^(-0.5) and x2 = +- sqrt(1-x1^2)
        (n1,n2,n3) = plane_normal
        x1 = n2 * np.power( (n1**2 + n2**2), -0.5 )
        x2 = math.sqrt(1 - x1**2)
        # Change sign of x2 if needed
        if abs(n1*x1 + n2*x2) > abs(n1*x1 + n2*x2*-1):
            x2 *= -1 
        # Check that first axis is really orthogonal to normal and normalized
        assert abs(n1*x1+n2*x2) < 1e-14  and abs(math.sqrt(x1**2 + x2**2) -1) < 1e-15, "Determination of first axis failed!"
        # Assemble first axis
        plane_axis_0 = np.array([x1, x2, 0])
        # Get second axis through cross product
        plane_axis_1 = np.cross(plane_normal, plane_axis_0)
        plane_axis_1 = plane_axis_1 / np.linalg.norm(plane_axis_1) # normalize
        
    return plane_axis_0, plane_axis_1
    
def ls_circle_1(points):
    """
    Least squares fit of circle to points, analogous to ls_sphere_3.

    Modified code following approach described in:
    https://jekel.me/2015/Least-Squares-Sphere-Fit/
    
    Approach:
    Starting point: (x-x0)^2 + (y-y0)^2  = r^2
    Rearranged to: x^2 + y^2 = 2x*x0 + 2y*y0 + 1*(r^2 -x0^2 -y0^2)
        --> Matrix:     f    =  A c
    with  f: nx1 vector with (xi^2 + yi^2) in i=1:n rows
          A: nx3 matrix with rows (2xi, 2yi, 1)
          c: 3x1 target matrix [x0, y0, (r^2 - x0^2 - y0^2)]

    Parameters
    ----------
    points : numpy.ndarray
        (n,2) array of 2d points.

    Returns
    -------
    center : numpy.ndarray
        Circle center.
    radius : float
        Circle radius.
    rmse : float
        Root mean square error of circle fit.

    """
    # Assemble the A matrix
    A = np.append(2*points, np.ones( (points.shape[0],1) ), axis=1 ) # (2xi, 2yi, 1)
    # Assemble f
    f = np.sum( np.power(points,2), axis=1) # x^2 + y^2 
    
    # Solve linear equation system Ac = f using least-squares algorithm
    c, residuals, rank, singval = np.linalg.lstsq(A,f)
    # Extract the center (first two values in c)
    center = c[0:2]
    # solve for the radius, c[3] = r^2 - x0^2 - y0^2
        # --> radius = sqrt( c[3] + x0^2 + y0^2)
    radius = math.sqrt( c[2] + np.sum( np.power(center,2) ) )
    
    # determine RMSE by comparing distances of points to center to the radius
    rmse = sphere_fit_rmse(points, center, radius)
    
    return center, radius, rmse

def generate_circle_points(radius, center, n_angles=180):
    """Generate evenly spaced points on a circle."""
    # generate angles
    u = np.linspace(0.0, 2.0 * np.pi, n_angles)
    # calculate points
    x = radius * np.cos(u) + center[0]
    y = radius * np.sin(u) + center[1]

    # combine x and y
    circle_points = np.c_[x,y]
    
    return circle_points

def transform_coords_plane2global(points, plane_center, plane_ax0, plane_ax1):
    """Transform point coordinates relative to a 3D plane into global coordinates."""
    points_tf = plane_center + np.array([ coord[0]*plane_ax0 + coord[1]*plane_ax1 for coord in np.atleast_2d(points) ])
    if points_tf.shape[0] == 1:
        points_tf = points_tf[0]
    return points_tf

def project_points_on_plane(points_orig, plane_center, plane_ax0, plane_ax1, output_coord_sys='plane'):
    """
    Project 3D points on a (3D) plane.
    
    Output is either in the plane coordinate system (output_coord_sys='plane') 
    or 'global' for global 3D coordinates.
    Plane axes: orthonormal, can be generated from plane normal using plane_axes.
    
    Parameters
    ----------
    points_orig : numpy.ndarray
        Input point array with shape (n,3).
    plane_center : numpy.ndarray
        Coordinates of point in plane taken as origin of plane coordinate system.
    plane_ax0 : numpy.ndarray
        (3,) vector of length=1 in plane, used as first axis of plane coordinate system.
    plane_ax1 : numpy.ndarray
        (3,) vector of length=1 in plane, orthogonal to plane_ax0. 
        Used as first axis of plane coordinate system..
    output_coord_sys : str, optional
        'plane' or 'global'. If 'plane', coordinates of projected points are 
        given relative to plane center and axes (2D). If 'global', global 
        3D coordinates are returned. The default is 'plane'.

    Returns
    -------
    plane_points : numpy.ndarray
        Array of projected points. (n,2) if output_coord_sys='plane', otherwise (n,3).

    """    
    # Generate vectors from plane origin to points
    v = points_orig - plane_center
    # Get coordinates of points with respect to plane axes
    v_axis0 = np.dot(v, plane_ax0)
    v_axis1 = np.dot(v, plane_ax1)
    # Combine coordinate arrays
    plane_points = np.c_[v_axis0, v_axis1]
    # Transform into global coordinate system if wished
    if output_coord_sys=='global':
        plane_points = transform_coords_plane2global(plane_points, plane_center, plane_ax0, plane_ax1)
        
    return plane_points


def circle_fit_3d(points, plane_center, plane_normal, n_plot_angles=180, 
                  return_rmse=False, return_tuple=False):
    """
    Generate a least-squares circle fit of points in or close to a plane.
    
    Returns radius and center as well as some points on the circle e.g. for plotting.

    Parameters
    ----------
    points : numpy.ndarray
        (n,3) point array.
    plane_center : numpy.ndarray
        Coordinates of plane point used as plane origin.
    plane_normal : numpy.ndarray
        Plane normal.
    n_plot_angles : int, optional
        Number of points in output fit circle. The default is 180.
    return_rmse : bool, optional
        If true, return circle fit error. The default is False.
    return_tuple : bool, optional
        If True, return results as a named tuple, always containing rmse. The default is False

    Returns
    -------
    circ_radius : float
        Radius of circle.
    circ_center_tf : numpy.array
        Circle center coordinates (global 3D coordinate system).
    circ_points_tf : numpy.array
        Circle points in global coordinate system, e.g. for plotting.
    circ_rmse : float, optional
        Root mean square error of circle fit.
    tuple_circle : named tuple, optional
        If return_tuple==True, function only returns the named tuple containing the above results.

    """
    # Step 1: Generate orthonormal axes in plane
    plane_ax0, plane_ax1 = plane_axes(plane_normal)
    
    # Step 2: Project points on the plane and get their coordinates with respect to the plane
    plane_points = project_points_on_plane(points, plane_center, plane_ax0, plane_ax1, output_coord_sys='plane')
    
    # Step 3: Do a least-squares circle fit of the points in the plane
    circ_center, circ_radius, circ_rmse = ls_circle_1(plane_points)
    
    # Step 4: For plotting purposes: return circle center and some circle points in global coordinate system
    circ_center_tf = transform_coords_plane2global(circ_center, plane_center, plane_ax0, plane_ax1) # Transform circle center
    
    circ_points = generate_circle_points(circ_radius, circ_center, n_angles=n_plot_angles) # Generate circle points
    circ_points_tf = transform_coords_plane2global(circ_points, plane_center, plane_ax0, plane_ax1) # Transform into global coord system
    
    if return_tuple:
        # Make a named tuple to store and return results
        tuple_circle = namedtuple('circle', ('radius', 'center', 'rmse', 'normal', 'points'))
        return tuple_circle(circ_radius, circ_center_tf, circ_rmse, plane_normal, circ_points_tf)
        
        
    if return_rmse:
        # RMSE should be the same in plane coordinates and global coordinates since there's no scaling
        return circ_radius, circ_center_tf, circ_points_tf, circ_rmse
    
    return circ_radius, circ_center_tf, circ_points_tf
        
    
def circle_fit_3d_points(points, n_plot_angles=720):    
    """
    Fit a circle into points (3D).

    Parameters
    ----------
    points : numpy.array
        Point array with shape (n,3).
    n_plot_angles : int, optional
        Number of points in output circle for plotting. The default is 720.

    Returns
    -------
    circle : named tuple
        Circle fit results including radius, center, rmse, plane normal and
        circle points for plotting.

    """
    # Get plane through points
    plane_center, (eigvec, eigval) = PCA(points)
    plane_normal = normalize_vector( eigvec[:,-1] )
    # Perform circle fit
    radius, center, circ_points, rmse = circle_fit_3d(points, plane_center, plane_normal, 
                                                      n_plot_angles, return_rmse=True)
    # Make a named tuple to store and return results
    tuple_circle = namedtuple('circle', ('radius', 'center', 'rmse', 'normal', 'points'))
    
    return tuple_circle(radius, center, rmse, plane_normal, circ_points)
    

#!!! Renamed and modified, needed for opening angle functions
def fit_plane_and_circle(points, normal_ref_point, N_to_ref=False, n_plot_angles=360):
    """Fit a plane and circle into given points, with a reference point to indicate normal direction.
    
    Results are returned as named tuples containing plane and circle parameters.
    """
    # Fit a plane
    res_plane = fit_plane(points, normal_ref_point, N_to_ref=N_to_ref) # Normal points outwards
    # Calculate distances and rmse to plane
    plane_dist = dist_to_plane_signed(points, res_plane)
    plane_rmse = np.linalg.norm(plane_dist) / np.sqrt(len(plane_dist))
    # Calculate positions of points after projecting them on the plane
    points_projected = points - np.matmul( np.atleast_2d( np.dot(points-res_plane.center, res_plane.normal) ).T, 
                                               np.atleast_2d(res_plane.normal) )
    tuple_plane_full = namedtuple('plane_fit', ['center', 'normal', 'rmse', 'point_distances', 'points_projected'])
    res_plane_full = tuple_plane_full(res_plane.center, res_plane.normal, plane_rmse, plane_dist, points_projected)
    
    # Fit the circle and save as named tuple
    res_circle = circle_fit_3d(points, res_plane.center, res_plane.normal, n_plot_angles=n_plot_angles, 
                                  return_tuple=True)
    
    return res_plane_full, res_circle   
    
    
    
    
    
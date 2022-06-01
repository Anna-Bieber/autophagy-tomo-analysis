#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for performing an ellipsoid fit.

Created on Fri Aug 28 15:45:40 2020

Functions were transcribed to python from MATLAB Code in github.com/pierre-weiss/FitEllipsoid (08/2020)
Reference paper: 
    Kovac B, Fehrenbach J, Guillaume L, Weiss P (2019) 
    FitEllipsoid: a fast supervised ellipsoid segmentation plugin. 
    BMC Bioinformatics 20: 142
    doi:10.1186/s12859-019-2673-0

@author: anbieber
"""

import numpy as np
from numpy.linalg import inv, eig
from math import sqrt
import pyvista as pv
import matplotlib.pyplot as plt

#%% helper functions

def project_to_simplex(y):
    """
    Project an n-dim vector y to the simplex Dn.
    
    Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    (c) Xiaojing Ye, 2011
    
    Algorithm is explained as in the linked document http://arxiv.org/abs/1101.6081
    """
    bget = False; 
    s = -np.sort(-y) # Workaround to get sorting in descending order
    
    tmpsum = 0
    for i, si in enumerate(s[:-1]):
        tmpsum += si
        tmax = (tmpsum-1)/(i+1)
        if tmax >= s[i+1]:
            bget = True
            break
    
    if ~bget:
        tmax = (tmpsum + s[-1] - 1) / len(y)
    
    x = np.maximum(y-tmax, 0) # element-wise comparison, takes larger value for each element
    
    return x

def project_on_B(q0):
    """Project q0 on B."""
    Q0 = np.array([[q0[0],          q0[3]/sqrt(2),  q0[4]/sqrt(2)],
                   [q0[3]/sqrt(2),  q0[1],          q0[5]/sqrt(2)],
                   [q0[4]/sqrt(2),  q0[5]/sqrt(2),  q0[2]]])
    [s0,U]=eig(Q0)
    s = project_to_simplex(s0)
    S = np.diag(s)
    Q = U @ S @ U.T
    
    q = np.concatenate( (np.array([Q[0,0], Q[1,1], Q[2,2], sqrt(2)* Q[1,0], sqrt(2)* Q[2,0], sqrt(2)* Q[2,1]]), 
                         q0[6:] ) )
    
    return q

#%% Main function



def fit_ellipsoid_DR_SVD(x, n_iter=1000):
    """
    Fit an ellipsoid to points.
    
    Given a set of points x=(x1,..,xn), this function finds a fitting
    ellipsoid in 3D, by using the approach proposed in the companion paper. 
    This method is not affine invariant. DR stands for Douglas-Rachford
    (Lions-Mercier would be more accurate).
    
    The output ellipsoid E is described implicitely by a triplet (A,b,c):
    E={x in R^3, <Ax,x> + <b,x> + c=0}
    or alternatively by vector q =(a11,a22,a33,sqrt(2)a12,sqrt(2)a13,sqrt(2)a23,b1,b2,b3,c).
    
    INPUT:
    - x: set of coordinates of size 2xn.
    - n_iter: number of iterations in Douglas-Rachford algorithm.
    OUTPUT:
    - A,b,c : matrix, vector, scalar describing the ellipsoid.
    - q: (a11,a22,a33,sqrt(2)a12,sqrt(2)a13,sqrt(2)a23,b1,b2,b3,c).
    - CF: Cost function wrt to iterations.
    """
    # Find SVD of x and change coordinates
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean # Center points around origin
    x_eval, x_evec = eig(x_centered.T @ x_centered) # Singular value decomposition
    
    P = np.diagflat(np.power(x_eval, -0.5)) @ x_evec.T # Transformation matrix for normalization
    
    x_norm = P @ x_centered.T # normalize x
    
    # Assemble matrix D
    D = np.vstack( (x_norm**2, 
                    sqrt(2)*x_norm[0,:]* x_norm[1,:],
                    sqrt(2)*x_norm[0,:]* x_norm[2,:],
                    sqrt(2)*x_norm[1,:]* x_norm[2,:],
                    x_norm,
                    np.ones_like(x_norm[0,:]) ) )
    
    K = D @ D.T
    
    # The objective is now to solve min <q,Kq>, Tr(Q)=1, Q>=0

    c = x_norm.mean(axis=1) # center after normalization
    
    r2 = x_norm.var(axis=1).sum()
    
    u = 1/3 * np.hstack( (1, 1, 1,
                          0, 0, 0,
                          -2*c,
                          (c**2).sum()-r2 ) )
    
    # And now go to the Douglas-Rachford (Lions-Mercier) iterative algorithm
    gamma = 10; # parameter gamma
    
    M = gamma*K + np.eye(K.shape[0])   
    p = u
    CF = np.zeros(n_iter+1)
    
    # Iterative solution
    for k in range(n_iter):
        q = project_on_B(p)
        CF[k] = 0.5* q @ K @ q.T
        
        (solution, res, rank, sing) = np.linalg.lstsq(M, 2*q-p, rcond=None) # np.linalg.lstsq corresponds to Matlab's mldivide (\)
        p += solution - q
    
    q = project_on_B(q)
    CF[-1] = 0.5* q @ K @ q.T
    
    A2 = np.array([[q[0],          q[3]/sqrt(2),  q[4]/sqrt(2)],
                   [q[3]/sqrt(2),  q[1],          q[5]/sqrt(2)],
                   [q[4]/sqrt(2),  q[5]/sqrt(2),  q[2]]])
    
    b2 = q[6:9]
    c2 = q[9]    
    
    # Go back to initial basis

    A = P.T @ A2 @ P
    b = -2* A @ x_mean.T + P.T @ b2.T
    c = (A2 @ P @ x_mean.T).T @ (P @ x_mean.T) - b2.T @ P @ x_mean.T + c2
    
    q = np.hstack( (np.diag(A),
                    sqrt(2)*A[1,0], sqrt(2)*A[2,0], sqrt(2)*A[2,1],
                    b, c) )
    
    q = q / np.sum(np.diag(A)) # normalization to stay on the simplex
    
    return A, b, c, q, CF

def ellipsoid_triple_to_algebraic(A,b,c):
    """Given A,b,c of an ellipsoid E={x in R^3, <Ax,x> + <b,x> + c=0}, returns the "algebraic form", i.e. a 4x4 matrix."""
    Amat = np.zeros((4,4))
    Amat[0:3,0:3] = A
    Amat[0:3,3] = b/2
    Amat[3,0:3] = b/2
    Amat[3,3] = c
    return Amat
    
    

def find_ellipsoid_parameters(A, b, c, verbose=False):
    """Given an ellipsoid E={x in R^3, <Ax,x> + <b,x> + c=0}, finds the center, radii and rotation matrix from A, b & c."""    
    # Determine the center
    # center = solution of A*x + b/2 = 0 -> -A*x = b/2
    center = np.linalg.lstsq(-1*A, 0.5*b)[0]
    if verbose:
        print('Ellipsoid center: {}'.format(center))
        
    # Determine the radii of the axes
    Amat = ellipsoid_triple_to_algebraic(A, b, c) # put ellipsoid parameters into 4x4 algebraic form
    # Transformation to center
    T = np.eye(4) # 4x4 unity matrix
    T[3,0:3] = center
    R = T @ Amat @ T.T # transform to center
    # Determine eigenvalues
    evals, evecs = eig(R[0:3,0:3] / -R[3,3])
    radii = np.power(abs(evals), -0.5) # radii = 1 / sqrt( abs(evals) )
    if verbose:
        print('Ellipsoid radii: {}'.format(radii))
        
    # Determine the rotation matrix
    rotmat = inv(evecs)
    
    return center, radii, rotmat

# Wrapper function
    
def fit_ellipsoid_iter(x, n_iter=5000, plot_CF=True, return_CF=False):
    """
    Fit an ellipsoid using the approach by Weiss et al.
    
    This is just a wrapper to connect the two main functions.
    """
    # Make the fit (with SVD normalization for faster convergence, see the paper SI)
    A, b, c, q, CF = fit_ellipsoid_DR_SVD(x, n_iter)
    
    # Optional: plot cost function
    if plot_CF:
        plt.figure()
        plt.plot(CF[:-1] - CF.min())
        plt.yscale('log')
        plt.xlabel('Iteration number k'); plt.ylabel('Cost function f1(q(k)) - min f1(q)')
        plt.show()
        
    # Calculate ellipsoid parameters from A, b, c
    center, radii, rotmat = find_ellipsoid_parameters(A, b, c)
    # residual of Cost function
    residual = CF[-1]
    
    if return_CF:
        return center, radii, rotmat, residual, CF
    else:
        return center, radii, rotmat, residual
    
def check_convergence_ellipsoid_fit(CF, n_iter, conv_frac=0.1, cutoff=1e-17):
    """Check if ellipsoid fit converged."""
    # Determine required convergence area
    conv_area_start = np.round(n_iter*(1-conv_frac)).astype(int)
    # Calculate cost function difference and check for cutoff
    CF_diff = CF - CF[-1]
    if np.max(CF_diff[conv_area_start:]) > cutoff:
        return False
    return True
    
    
#%% Functions to generate cylinders representing the ellipsoid axes


def generate_axes_cylinders_half(center, axes, rotmat, cyl_radius=0.05):
    """Generate cylinders representing the half axes of an ellipsoid."""
    cylinders = {}
    # Loop over x,y,z
    for n, axis in zip(range(3), ('x','y','z')):
        # generate outer point on axis
        ax_point = np.zeros(3)
        ax_point[n] += axes[n] # add axis length
        ax_point = np.dot(ax_point, rotmat) # rotate by rotation matrix
        ax_point += center # shift by center
        # generate cylinder
        cylinders[axis] = pv.Cylinder(center=0.5*(center+ax_point), 
                                         direction = ax_point-center,
                                         radius=cyl_radius,
                                         height=axes[n])
    return cylinders

def generate_axes_cylinders(center, axes, rotmat, cyl_radius=0.05):
    """Generate cylinders representing the half axes of an ellipsoid."""
    cylinders = {}
    # Loop over x,y,z
    for n, axis in zip(range(3), ('x','y','z')):
        # generate outer point on axis
        ax_point = np.zeros(3)
        ax_point[n] += axes[n] # add axis length
        ax_point = np.dot(ax_point, rotmat) # rotate by rotation matrix
        ax_point += center # shift by center
        # generate cylinder
        cylinders[axis] = pv.Cylinder(center=center, 
                                         direction = ax_point-center,
                                         radius=cyl_radius,
                                         height=2*axes[n])
    return cylinders    

    
    
    
    
    
    
    
    
    
    
    
    
    
    




# Collection of python functions for tomo analysis

To use from a different folder, add this folder to your path:
```python
import sys
sys.path.insert(0, path_to_utilities)
```

## Script Files

### utility_functions
A collection of utility functions needed in all other scripts. Includes functions for:
* Loading files
* Vector operations (normalize, calculate angles or rotations)
* Array operations
* Point cloud operations (PCA, finding outliers, checking whether points are in a convex hull, cleaning by binary opening)
* dictionaries: comparing two dictionaries, turning all named tuples within a dict into dictionaries

Dependencies: numpy, scipy, sklearn, tifffile, pyvista, pyvistaqt, collections


### distance_functions
Different functions to calculate distances between points, point clouds and meshes. <br />
Dependencies:
* numpy, scipy.spatial, collections
* fitting_functions, utility_functions

### fit_ellipsoid_weiss
Ellipsoid fitting functions adapted from the package [FitEllipsoid](https://github.com/pierre-weiss/FitEllipsoid), originally written in Matlab, into Python. Reference: [Kovac et al.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2673-0), *BMC Bioinformatics* 2019.

### fitting_functions
Functions for fitting different geometric shapes into 3D point clouds, and for analyzing the result. Included are functions for fitting:
* Lines and spline curves
* Planes
* 2D polynomial or spline surfaces
* Circles
* Spheres (least-squares fits)
* Ellipsoids (least-squares fit from [this source](http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html), however, the iterative algorithm in fit_ellipsoid_weiss is preferred especially for noisy or incomplete data.)

Additionally, there are functions to:
* Project points into planes, and transfer points between different coordinate systems
* Create sphere or ellipsoid meshes
* Estimate the minimum distance of a point to an ellipsoid, and use this to calculate the rmse of an ellipsoid fit.

Dependencies:
* numpy, scipy, math, pyvista, collections, functools, tqdm, itertools
* utility_functions, mesh_functions

## histogram_functions
Functions for binning and making 1D and 2D histograms from arrays and meshes. <br />
Dependencies:
* numpy, scipy, collections, matplotlib
* utility_functions

## mesh_functions
Functions to manipulate pyvista meshes, e.g.
* extracting specific cells
* constructing new meshes
* finding mesh borders and removing all cells within a certain distance to the border
* making neighbor lists
* refining normals to be more consistent between neighboring cells
* removing very small cells without leaving a holes
* detecting and filling holes in a mesh

Dependencies:
* numpy, scipy, matplotlib, [**pyvista**](https://docs.pyvista.org/), pyvistaqt
* utility_functions

## opening_angle_functions
Functions needed to determine the opening angle of a phagophore (see the [biorxiv preprint](https://doi.org/10.1101/2022.05.02.490291) for details). For usage, check the notebook Run_phagophore_analysis.ipynb. <br />
Dependencies:
* numpy, scipy, collections
* utility_functions, fitting_functions

## reference_mesh_functions
Functions needed for constructing an idealized reference rim from a real-world phagophore rim. For usage, see the notebook Make_rim_reference_mesh.ipynb. <br />
Dependencies:
* numpy, scipy, pyvista, functools, time
* utility_functions, mesh_functions, fitting_functions

## contact_analysis_functions
Specific functions to analyze membrane peaks at contact sites (in this case, peaks in the outer phagophore membrane towards the vacuole). <br />
Dependencies:
* numpy, scipy, pyvista
* fitting_functions, rim_analysis_functions, histogram_functions

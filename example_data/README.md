# Example data from tomograms

Exemplary segmentation data used as input files in the notebooks, and result files from running the notebooks that can be used to verify that notebooks are running correctly. Segmentations were done on denoised tomograms (cryoCARE) at bin 4 (pixel size 1.408 nm) using TomoSegMemTV and Amira, and always mark the middle of the respective membranes.

## Folders and files
* autophagosome_analysis:
  * AP_middle_labels.tif: Rough segmentation of an autophagosome (membrane middles).
  * Notes.txt: Info on tomogram and segmentation.
  * Autophagosome_results.npy: Output of the notebook Run_autophagosome_analysis.ipynb


* phagophore_analysis:
  * Phagophore_middle_labels.tif: Rough segmentation of a phagophore (membrane middles).
  * Notes.txt: Info on tomogram and segmentation.
  * Phagophore_results.npy: Output of the notebook Run_phagophore_analysis.ipynb

* phagophore_vacuole_contact:
  * Phagophore_mesh.vtp: Mesh of a phagophore, generated with [PyCurv](https://github.com/kalemaria/pycurv) from the segmentation of the rim. The curvature determination was performed with a radius hit of 8 nm.
  * vacuole_contact_labels.tif: Segmentation of the vacuole next to the phagophore.

* rim_analysis:
  * Ph_rim0_.AVV_rh8.vtp: Mesh of a phagophore rim, generated with [PyCurv](https://github.com/kalemaria/pycurv) from the segmentation of the rim. The curvature determination was performed with a radius hit of 8 nm.
  * Notes.txt: Info on tomogram for internal bookkeeping
  * Ph_rim0_results.npy: Output of the notebook Run_rim_analysis.ipynb, containing results of the rim analysis.
  * rim_mesh_out.vtp: Rim mesh with additional value arrays determined in Run_rim_analysis.ipynb.
  * mid_surf_mesh_out.vtp: Mesh of the middle surface between the two sides of the rim, generated in Run_rim_analysis.ipynb.
  * rim_reference_mesh.vtp: Output mesh of Make_rim_reference_mesh.ipynb. An idealized rim with the same overall shape, rim length, area and intermembrane distance as the experimental rim, but without swelling or constriction.

* wrapped_AB_analysis:
  * wrapped_AB_middle_labels.tif: Rough segmentation of an autophagic body, partially wrapped by the vacuole membrane (membrane middles).
  * Notes.txt: Info on tomogram and segmentation.


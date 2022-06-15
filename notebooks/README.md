# Jupyter notebooks showing analysis of autophagic structures on example example_data

## Run_autophagosome_analysis.ipynb
Analysis of roughly segmented autophagosome membranes, including intermembrane distance analysis, sphere and ellipsoid fits, and an estimation of the area to lumen ratio.

## Run_phagophore_analysis.ipynb
Analysis of roughly segmented phagophore membranes, including intermembrane distance analysis, sphere and ellipsoid fits, and an estimation of the completeness based on the opening angle (see the [biorxiv preprint](https://doi.org/10.1101/2022.05.02.490291) for details.)

## Run_rim_analysis.ipynb
Analysis of a phagophore rim mesh after running [PyCurv](https://github.com/kalemaria/pycurv). Includes an automated separation of rim tip and the two sides, positional mapping of intermembrane distance and curvature and an analysis of rim dilation.

## Make_rim_reference_mesh
Based on the experimental example rim, construct an idealized rim with the same overall shape, rim length, area and intermembrane distance as the experimental rim, but without swelling or constriction.

## Run_wrapped_AB_analysis.ipynb
Analyze the intermembrane distance of the part of an autophagic body that is still wrapped by the vacuole.

## Run_vacuole_contact_analysis.ipynb
Analyze a peak in the outer membrane of a phagophore towards the vacuole. Extract the peak from the phagophore mesh, measure its dimensions and make a 2d map of the peak elevation and distance to the vacuole.

# autophagy-tomo-analysis
Scripts for analysis of autophagy-related structures in cryo-electron tomography data.

## Description
Using membrane segmentations from cellular tomograms as input data, the scripts provided here can be used to calculate intermembrane distances of double membranes, find the best-fitting spheres and ellipsoids and derived parameters, and analyze various parameters associated with meshes. The code was written specifically for the analysis of autophagosomes and phagophores but many concepts should be transferable to other cellular structures as well.


### Folders
* **utilities**: collection of all custom functions used in this project.
* **notebooks**: Jupyter notebooks showcasing the use of the custom functions in analyzing autophagosomes, phagophores and phagophore rims, wrapped autophagic bodies and membrane peaks at phagophore-vacuole contact sites.
* **example_data**: Example segmentation data of autophagic structures from tomograms of nitrogen-starved yeast cells, used as input in the notebooks.



## Citation
These scripts were developed as part of the study:

*In situ structural analysis reveals membrane shape transitions during autophagosome formation*

Anna Bieber,  Cristina Capitanio,  Philipp S. Erdmann, Fabian Fiedler, Florian Beck, Chia-Wei Lee, Delong Li,  Gerhard Hummer,  Brenda A. Schulman,  Wolfgang Baumeister and Florian Wilfling

PNAS 2022, vol. 119 no. 39, https://doi.org/10.1073/pnas.2209823119

biorxiv doi: https://doi.org/10.1101/2022.05.02.490291

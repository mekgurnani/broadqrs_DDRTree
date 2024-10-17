# Applying DDRTree to the broad QRS complex

This repository contains the code to run the DDRTree algorigthm on VAE-features derived from ECGs described in our paper "**Revisiting abnormalities of ventricular depolarisation: Redefining phenotypes and associated outcomes using tree-based dimensionality reduction**". The paper explores the derived tree variables (tree dimensions and phenogroups) across clinical variables, disease outcomes, CRT response and echocardiography measures. 

## Directory structure 

The provided code is to run the DDRTree algorithm, apply hierarchical clustering to form representative branches and to project other populations onto the tree. Additionally, the code to host the Dash Plotly interactive plot showing the derived DDRTree from this work is also included in this repository. 

```
├── ddrtree_analysis_script/                            # Contains scripts for the main analysis 
│   ├── make_ddrtree.R                                  # Main script to run DDRTree algorithm
│   ├── DDRTree_hier_clustering.R                       # Script to apply hierarchical clustering to sub-branches
│   ├── projecting_ext_val_data.ipynb                   # Jupyter Notebook with code to predict tree variables for new dataset
│   └── dist_est.py                                     # Script to apply distance estimating algorithm required in projecting_ext_val_data.ipynb
├── src/                                                # Source code for the Dash Plotly app
│   ├── MODEL_INFO.json                                 # Configuration of VAE model
│   ├── app.py                                          # Main script to run the app
│   ├── group28_medians_3d.npy                          # Median ECG data - medians
│   ├── group28_stds_3d.npy                             # Median ECG data - standard deviations
│   ├── tree_proj_full_branches_plotly_relevant.csv     # Data needed to plot the tree (phenogroup assignments/tree dimensions)
│   ├── val_i_factors.json                              # Configuration of VAE model
│   └── assets/                                         # Styling the app
│       └── style.css                                   # Styling for the app
├── render.yaml                                         # Configuration of the Dash Plotly app
├── requirements.txt                                    # List of dependencies for running the Dash Plotly app
└── README.md                                           # This README file
```

The interactive Dash Plotly app can be found [here](https://broadqrs-ddrtree-viz.onrender.com).

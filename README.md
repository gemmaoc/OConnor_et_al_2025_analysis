# OConnor_et_al_2025_analysis

ABOUT
-----
This repo contains code used for analysis and figures in O'Connor et al., 2025 (https://doi.org/10.1038/s41561-025-01757-6).

Contents include code for formatting and analyzing output from regional MITgcm ocean simulations in the Amundsen Sea and adjacent regions. The model was forced by repeated atmospheric forcings for 5 years, revealing the oceanic and ice-shelf-melt response to various cumulative atmospheric conditions. Output from the simulations were compared to proxy-constrained reconstructions of atmospheric circulation over the 20th century, for comparison to real historical conditions near West Antarctica.

The results reveal repeated northerly wind anomalies, interacting with coastal polynyas, as the primary driver of ice-shelf melting in the Amundsen Sea Embayment of West Antarctica. The proxy reconstructions reveal a significant northerly wind trend in this region over the 20th century, suggesting that a historical strengthening of northerly winds can explain the initiation of rapid ice-shelf melting in this region.

HOW TO USE
----------
The Jupyter notebooks used to generate the figures in O'Connor et al., 2025 are named after the respective figures in the article. Functions used for loading the simulations are contained in "Functions_load_output.py". Functions called for analysis are found in "Functions_data_analysis.py". Functions called for generating plots are found in "Functions_plotting.py". Metadata about the datasets used throughout this repo are found in "Datasets.py".

RESOURCES
---------
More details on the modeling setup and results can be found in O'Connor et al., 2025. 

Data:
- Output from the simulations and the proxy reconstructions are archived on Zenodo at https://zenodo.org/records/15243743. 
- Visualizations of the proxy reconstructions can be found on a web app developed by my undergraduate research assistant Jordan Tucker at
https://jortuck.github.io/PaleoclimateVisualizer/.

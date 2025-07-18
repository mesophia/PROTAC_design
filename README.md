# PROTAC_design
After generation of ternary compplexes or projected ternary complex from protein-protein complexes (POI-VHL) please follow the given order of running calculatiions:
1. "covert_chain_id.py"
2. "PPD_clustering.py"
3. "Lys_distance.py"
4. "Lys_excel_count.py"

Note: before running Lys_distance.py file, you must keep the "8RX0_prepared.pdb" file in same directory as the .py file

Principal Component Analysis (PCA), Time-lagged Independent Component Analysis (TICA), Implied Timescales (ITS), and Markov State Model (MSM) analyses could be performed following molecular dynamics simulations. These analyses are carried out using PyEMMA version 2.5, which is compatible with Python 2. Due to compatibility issues with Python 3, a dedicated environment can be created using the command:
conda create -n pyemma-env python=2.7

And then, please install PyEMMA within this environment.

The corresponding analysis scripts are:
1. "PCA_pyemma.py" – for PCA calculation
2. "TICA_pyemma.py" – for TICA analysis
3. "ITS_pyemma.py" – for implied timescale computation
4. "MSM_pyemma.py" – for MSM construction and analysis

These scripts collectively enabled the extraction of dominant conformational features and kinetic states from the simulation trajectories.

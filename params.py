"""
Define the parameters for main.py
"""

loadDirectory = "/data/tract2020/HCP_bundle_mapping/" # Folder with all subjects
saveDirectory = "/data/lfpjacobs"                     # Folder to which to save data, results, and model checkpoints

bundle = "CST_left" # BOI
n = 100             # Number of points on tracts
r_neighbor = 0.5    # Representation of neighboring tracts during training
n_center = 10       # Number of centers of mass in BOI

train = 6   # Number of training subjects
test = 5    # Number of testing subjects
val = 2     # Number of validation subjects
pseudo = 5  # Number of pseudo testing subjects for FP mining
n_tot = 105 # Total number of subjects

save = False    # Save the data for later runs
load = True     # Load from saved data from previous run 
t_SNE = False   # Save t-SNE embedding of the best epoch of the first validation subject
SVM = False     # Test the effect of an SVM on top of the network
FPMining = True # Test the effect of FP mining 

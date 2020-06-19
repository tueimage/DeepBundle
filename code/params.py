"""
Define the parameters for the experiments (model parameters can be changed in main.py)
"""

loadDirectory = "/data/tract2020/HCP_bundle_mapping/" # Folder with all subjects (HCP)
saveDirectory = "/data/lfpjacobs"                     # Folder to which to save data, results, and model checkpoints

bundle = "CST_left" # Bundle of interest (BOI)
n = 100             # Number of points on tracts
r_neighbor = 0.5    # Representation of neighboring tracts during training
n_center = 10       # Number of centers of mass in BOI used to define neighboring region

train = 6   # Number of training subjects
test = 5    # Number of testing subjects
val = 2     # Number of validation subjects
pseudo = 5  # Number of pseudo testing subjects for FP mining
n_tot = 105 # Total number of subjects

save = False     # Save the loaded data for later runs
load = True      # Load from saved data from previous run 
t_SNE = False    # Save t-SNE embedding of the best epoch of the first validation subject
SVM = False      # Test the effect of an SVM on top of the network
FPMining = False # Test the effect of FP mining 

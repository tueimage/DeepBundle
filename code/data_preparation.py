import random
import sys
import os
import scipy
import numpy as np
from nibabel import trackvis
from dipy.segment.metric import CenterOfMassFeature
from coarsening import coarsen
from coarsening import perm_data


def sample_tracts(tract, n, smoothing):
    """
    Return a sampled version of a tract
    Input:
    tract - The tract to be resampled
    n - The number of points sampled on each tract
    smoothing - The amount of smoothness
    """
    tck, _ = scipy.interpolate.splprep([tract[:, 0], tract[:, 1], tract[:, 2]], s=smoothing)
    u_fine = np.linspace(0, 1, n)
    x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)

    return x_fine, y_fine, z_fine

def first_coarsen(node_features, n):
    """
    Returns a graph object based on the coordinates of the tract
    Input:
    node_features - An array (n x 3) with the 3D coordinates of each point
    n - The number of points sampled on each tract
    """
    row = np.array(list(range(n-1))+list(range(1, n))) # row[i] connects to col[i]
    col = np.array(list(range(1, n))+list(range(n-1)))
    data = np.array([float(1) for i in range(2*(n-1))]) # Unit weights
    A = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n)).astype(np.float32)
    coarsening_levels = 2
    L, perm = coarsen(A, coarsening_levels)
    vert = perm_data(node_features.transpose(), perm)

    return vert.transpose().astype('float16'), L, perm

def coarsen_again(node_features, perm):
    """No need to compute the same Laplacians for each graph"""
    vert = perm_data(node_features.transpose(), perm)

    return vert.transpose().astype('float16')

def calc_dist_3d(p0, p1):
    """Return the Euclidean distance between points p0 and p1"""
    x0, y0, z0 = p0[0], p0[1], p0[2]
    x1, y1, z1 = p1[0], p1[1], p1[2]
    dist = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)

    return dist

def data_prep(directory, bundle, n, n_center, r_neighbor, datatype):
    """
    Returns a lists of vertices, corresponding labels (1 for bundle of interest, 0 for others),
    and the multi-level laplacians for training or a list of distances of the tracts to the BOI
    for validation or testing
    Input:
    directory - Folder with the 72 bundles of a subject
    bundle - Which bundle is the BOI
    n - The number of points sampled on each tract
    n_center - Number of centers of mass which determine the neighborhood for training data
    r_neighbor - The maximum ratio you want the neighboorhood tracts to represent during training
    datatype - "training" or "test_or_val", test_or_val returns graphdata and labels for all
                tracts of the subject, whereas training only returns the bundle of interest,
                neighbouring tracts (depending on r_neighbor), and randomly sampled tracts from
                other bundles to create a balanced training set
    """
    data_all, points, center_list = [], [], []
    FirstIteration1, FirstIteration2 = True, True
    for filename in os.listdir(directory): # Loop through bundels
        if filename == (bundle+".trk"): # Only look at BOI
            trk_path = os.path.join(directory, filename)
            streams, _ = trackvis.read(trk_path)
            streamlines = [s[0] for s in streams] # List of tracts
            for tract in streamlines:
                x_fine, y_fine, z_fine = sample_tracts(tract, n, 2)
                vertices = np.hstack([x_fine.reshape(x_fine.shape[0], 1),
                                      y_fine.reshape(y_fine.shape[0], 1),
                                      z_fine.reshape(z_fine.shape[0], 1)])
                if FirstIteration1:
                    data, L, perm = first_coarsen(vertices, n)
                    FirstIteration1 = False
                else:
                    data = coarsen_again(vertices, perm)
                data_all.append((data, 1))
                points_loc = [int(j) for j in np.linspace(0, n-1, n_center)]
                points.append([vertices[i, :] for i in points_loc]) # Centroid(s) per tract
            feature = CenterOfMassFeature()
            if FirstIteration2: # First tracts determines orientation for the rest of the bundle
                ref = points[0][int(n_center/4)] # Look at point between the end and the median
                FirstIteration2 = False
            centroids_flipped = []
            for i in points:
                first_point = i[int(n_center/4)]
                last_point = i[-int(n_center/4)-1]
                if calc_dist_3d(ref, last_point) < calc_dist_3d(ref, first_point):
                    i = list(reversed(i)) # Flip points if the tract is in the opposite direction
                centroids_flipped.append(i)
            for j in range(n_center):
                points = [k[j] for k in centroids_flipped]
                center = list(map(feature.extract, [np.array(points)]))[0][0] # Center of mass
                center_list.append(center)
            if datatype == "training":
                max_distances = []
                for l in range(len(center_list)):
                    if l == 0:
                        dist = calc_dist_3d(center_list[l], center_list[l+1]) # One end
                    elif l == n_center-1:
                        dist = calc_dist_3d(center_list[l], center_list[l-1]) # Other end
                    else:
                        dist = max([calc_dist_3d(center_list[l], center_list[l+1]),
                                    calc_dist_3d(center_list[l], center_list[l-1])]) # In between
                    max_distances.append(dist)
            break

    # No need for more negative data than positive, so limit number of negative labels
    n_tracts_neg = 0
    if datatype == "training":
        for filename in os.listdir(directory):
            if filename != (bundle+".trk"): # Every bundle but the BOI
                trk_path = os.path.join(directory, filename)
                streams, _ = trackvis.read(trk_path)
                n_tracts_neg += len(streams)
        neg_id_list = random.sample(range(n_tracts_neg), 2*len(data_all)) # Times 2 to be sure tracts in the neighboorhood will not be repeated and still have a balanced training set

    neg_id = 0
    data_near, data_rest, min_dist_l = [], [], []
    for filename in os.listdir(directory):
        trk_path = os.path.join(directory, filename)
        streams, _ = trackvis.read(trk_path)
        streamlines = [s[0] for s in streams]
        if filename != (bundle+".trk"): # Every bundle but the BOI
            tract_id = 0 # Initialize counting
            tract_id_list = []
            for tract in streamlines:
                x_fine, y_fine, z_fine = sample_tracts(tract, n, 2)
                vertices = np.hstack([x_fine.reshape(x_fine.shape[0], 1),
                                      y_fine.reshape(y_fine.shape[0], 1),
                                      z_fine.reshape(z_fine.shape[0], 1)])
                if datatype == "training": # Add neighbouring tracts
                    nrpoints = 0 # Initialize counting of number of points in the neighborhood
                    for point in vertices:
                        list_dist = []
                        for p in center_list:
                            list_dist.append(calc_dist_3d(point, p))
                        FirstIteration3 = True
                        for i in range(len(center_list)):
                            if list_dist[i] < max_distances[i] and FirstIteration3: # In neighborhood
                                FirstIteration3 = False # Do not count the same point twice
                                nrpoints += 1
                        if nrpoints == n/2: # If half of the points lay in the neighborhood
                            data = coarsen_again(vertices, perm)
                            data_near.append((data, 0))
                            if r_neighbor != 0:
                                tract_id_list.append(tract_id)
                            break

                elif datatype == "test_or_val": # Add every tract for testing or validation dataset
                    min_dist = float("inf") # Initialize minimum distance
                    data = coarsen_again(vertices, perm)
                    data_all.append((data, 0))
                    feature = CenterOfMassFeature()
                    c = list(map(feature.extract, [vertices]))[0][0] # Center of mass of tract
                    for p in center_list:
                        d = calc_dist_3d(c, p) # Distance to one of the centers of mass of the BOI
                        if d < min_dist:
                            min_dist = d
                    min_dist_l.append(min_dist)
                else:
                    sys.exit("Please enter a valid datatype ('training' or 'test_or_val')")
                tract_id += 1

            if datatype == "training": # Add additional random tracts
                tract_id2 = 0 # Initialize a second counting
                extra_tracts = [x for x in range(len(streamlines)) if x not in tract_id_list] # Do not consider tracts that are already in the neighbourhood
                for tract in streamlines:
                    if tract_id2 in extra_tracts and neg_id in neg_id_list:
                        x_fine, y_fine, z_fine = sample_tracts(tract, n, 2)
                        vertices = np.hstack([x_fine.reshape(x_fine.shape[0], 1),
                                              y_fine.reshape(y_fine.shape[0], 1),
                                              z_fine.reshape(z_fine.shape[0], 1)])
                        data = coarsen_again(vertices, perm)
                        data_rest.append((data, 0))
                    tract_id2 += 1 # Count extra tracts
                    neg_id += 1 # Count tracts (all bundles)

    if datatype == "training": # Sample for a balanced training set
        if len(data_near) > int(r_neighbor*len(data_all)):
            data_neg = random.sample(data_near, int(r_neighbor*len(data_all))) + \
            random.sample(data_rest, int((1-r_neighbor)*len(data_all)))
        else:
            data_neg = data_near + random.sample(data_rest, len(data_all)-len(data_near)) # Fill the rest with random tracts if necassary
        data_all.extend(data_neg)

    X, y = zip(*data_all) # Seperate vertices and labels
    if datatype == "training":
        return list(X), list(y), L
    return list(X), list(y), min_dist_l
    
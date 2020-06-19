import random
import os
import statistics
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
import model
import params
from data_preparation import data_prep


## LOAD DATA
def load_data(save, path_save, path_load):
    id_all = random.sample(range(params.n_tot), params.train+params.test+params.val+params.pseudo) # Random subjects
    #id_all = [92, 97, 7, 94, 21, 102, 75, 56, 53, 39, 49, 54, 85, 70, 6, 81, 69, 42] # Fixed for experiments
    id_train = id_all[:params.train]
    id_test = id_all[params.train : params.train+params.test]
    id_val = id_all[params.train+params.test : params.train+params.test+params.val]
    id_pseudo = id_all[params.train+params.test+params.val:]
    idx = 0
    Xtrain, ytrain, Xtest_subj, ytest_subj, Xtest, ytest, Xval, yval, Xpseudo, ypseudo, min_dist_l = ([] for i in range(11))
    for filename in os.listdir(path_load):
        if filename not in ("License.txt", "Readme.txt"):
            subj = os.path.join(path_load, filename)
            subj_dir = subj + "/tracts"
            if idx in id_train:
                print("Training: {}".format(filename))
                vert, y, L = data_prep(subj_dir, params.bundle, params.n, params.n_center, params.r_neighbor, "training")
                Xtrain.extend(vert) # Combine all subjects
                ytrain.extend(y)
            if idx in id_test:
                print("Testing: {}".format(filename))
                vert, y, _ = data_prep(subj_dir, params.bundle, params.n, params.n_center, params.r_neighbor, "test_or_val")
                Xtest_subj = np.array(vert) # Keep subjects seperated for testing
                ytest_subj = np.array(y)
                Xtest.append(Xtest_subj)
                ytest.append(ytest_subj)
            if idx in id_val:
                print("Validation: {}".format(filename))
                vert, y, min_dist = data_prep(subj_dir, params.bundle, params.n, params.n_center, params.r_neighbor, "test_or_val")
                Xval_subj = np.array(vert) # Keep subjects seperated (for now) for validation
                yval_subj = np.array(y)
                Xval.append(Xval_subj)
                yval.append(yval_subj)
                min_dist_l.append((filename, min_dist))
            if idx in id_pseudo:
                print("Pseudo testing: {}".format(filename))
                vert, y, _ = data_prep(subj_dir, params.bundle, params.n, params.n_center, params.r_neighbor, "test_or_val")
                Xpseudo_subj = np.array(vert) # Keep subjects seperated for pseudo testing
                ypseudo_subj = np.array(y)
                Xpseudo.append(Xpseudo_subj)
                ypseudo.append(ypseudo_subj)
            idx += 1

    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)

    if save:
        with open(path_save+'/Data_{}_{}'.format(params.bundle, params.r_neighbor), 'wb') as f:
            pkl.dump((Xtrain, ytrain, Xtest, ytest, Xval, yval, Xpseudo, ypseudo, min_dist_l, L), f, protocol=4)

    return Xtrain, ytrain, Xtest, ytest, Xval, yval, Xpseudo, ypseudo, min_dist_l, L


if params.load:
    with open(params.saveDirectory+'/Data_{}_{}'.format(params.bundle, params.r_neighbor), 'rb') as f:
        Xtrain, ytrain, Xtest, ytest, Xval, yval, Xpseudo, ypseudo, min_dist_l, L = pkl.load(f)
else:
    Xtrain, ytrain, Xtest, ytest, Xval, yval, _, _, min_dist_l, L = load_data(params.save, params.saveDirectory, params.loadDirectory)


## MODEL PARAMETERS
params1 = dict(
    dir_name = 'GCNNparameters',       # Directory name for results
    model_save = params.saveDirectory, # Path to save model checkpoints
    num_epochs = 30,                   # Number of epochs for training
    batch_size = 128,                  # Batch size for training

    F = [32, 64],                      # Number of graph convolutional filters
    K = [20, 20],                      # Polynomial orders
    p = [2, 2],                        # Pooling sizes
    M = [512, 2],                      # Output dimensionality of fully connected layers

    regularization = 2e-5,             # L2 regularization
    learning_rate = 1e-4,              # Learning rate for training
    decay_rate = 0.95,                 # Decay rate
    decay_steps = int(0.1*30*Xtrain.shape[0] / 128),        # Decays 10x with decay_rate
    imbalance = list(ytrain).count(0)/list(ytrain).count(1) # Should be balanced with 1
)
model1 = model.cgcnn(L, **params1)

## TRAIN MODEL AND SHOW PERFORMANCE
loss_train, loss_val, F1_val, intermediate_val_list, epochs, intermediate_train, epoch_tsne, prec_val, rec_val = model1.fit(Xtrain, ytrain, Xval, yval, params.t_SNE, False)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(epochs, loss_train, 'k.-', label='Training loss')
ax1.plot(epochs, loss_val, 'k.--', label='Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(epochs, F1_val, 'b.-', label='Validation F1-score')
ax2.set_ylabel('F1-score', color='b')
ax2.legend()
plt.savefig(params.saveDirectory+'/training.png')
plt.show()
print("Validation \n Precision: {} \n Recall: {}".format(prec_val, rec_val))

inter_test1 = []
with open(params.saveDirectory+'/validation.txt', 'w') as f:
    print("Precision: {}, recall: {}".format(prec_val, rec_val), file=f)

prec_l, rec_l, f1_l = [], [], []
for i in range(len(Xtest)): # Per subject
    _, f1, prec, rec, inter = model1.evaluate(Xtest[i], ytest[i], False)
    feat_test = np.vstack(inter)[:ytest[i].shape[0], :]
    inter_test1.append(feat_test)
    prec_l.append(prec)
    rec_l.append(rec)
    f1_l.append(f1)

print("Testing \n Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                           statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                           statistics.mean(f1_l), statistics.stdev(f1_l)))

with open(params.saveDirectory+'/testing.txt', 'w') as f:
    print("Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                    statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                    statistics.mean(f1_l), statistics.stdev(f1_l)), file=f)


## SAVE t-SNE EMBEDDING OF THE FIRST VALIDATION SUBJECT OF THE BEST EPOCH
if params.t_SNE:
    labels = yval[0]
    idx1 = random.sample(range(np.count_nonzero(labels == 0)), 500) # 500 tracts outside the BOI
    idx2 = random.sample(range(np.count_nonzero(labels == 1)), 100) # 100 tracts in the BOI
    distances_s = np.array([0.125*d for d in min_dist_l[0][1]])[idx1] # 0.125 cm^3 voxel dimension

    tsne = sklearn.manifold.TSNE(n_components=2, random_state=42, perplexity=100, init='pca')
    intermediates = np.concatenate(intermediate_val_list, axis=0)[:labels.shape[0], :]
    inter_pos = intermediates[labels == 1][idx2]
    inter_neg_s = intermediates[labels == 0][idx1]
    intermediates_tsne = tsne.fit_transform(np.vstack((inter_pos, inter_neg_s)))

    plt.figure(figsize=(8, 8))
    plt.scatter(intermediates_tsne[100:, 0], intermediates_tsne[100:, 1], c=distances_s)
    cb = plt.colorbar()
    cb.set_label('Shortest euclidian distance to BOI [cm]')
    plt.scatter(intermediates_tsne[:100, 0], intermediates_tsne[:100, 1], c='red')
    plt.box(on=None)
    plt.axis('off')
    plt.title('t-SNE GCNN of subject #{} (epoch: {})'.format(min_dist_l[0][0], int(epoch_tsne)+1))
    plt.savefig(params.saveDirectory+"/epoch{}.png".format(int(epoch_tsne)+1))


## USE AN SVM ON TOP OF THE NETWORK
if params.SVM:
    model2 = SVC(kernel="poly", degree=3, C=1e5, gamma='scale')
    features = np.vstack(intermediate_train)[:ytrain.shape[0], :] # Training features
    model2.fit(features, ytrain) # Train the SVM

    prec_l, rec_l, f1_l = [], [], []
    for i in range(len(Xtest)):
        feat = inter_test1[i]
        feat[feat >= 1e308] = 1e308 # float64 limit
        feat[np.isnan(feat)] = 0
        predicted_class = model2.predict(feat)
        labels = ytest[i]

        prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, predicted_class, average='binary')
        prec_l.append(prec)
        rec_l.append(rec)
        f1_l.append(f1)

    print("SVMmodel: \n Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                                 statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                                 statistics.mean(f1_l), statistics.stdev(f1_l)))
    with open(params.saveDirectory+'/SVM_results.txt', 'w') as f:
        print("SVMmodel: \n Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                                     statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                                     statistics.mean(f1_l), statistics.stdev(f1_l)), file=f)


## ADD FP MINING
if params.FPMining:
    FPs = []
    for i in range(len(Xpseudo)): # Per subject
        FPidx = model1.evaluate(Xpseudo[i], ypseudo[i], True)
        FPs_subj = Xpseudo[i][FPidx, :, :]
        FPs.append(FPs_subj)

    FPs = np.concatenate(FPs, 0) # FPs of all subjects

    Xtrain_added = np.concatenate([Xtrain, FPs], 0)
    ytrain_added = np.concatenate([ytrain, np.zeros((FPs.shape[0], 1)).ravel()], 0)

    params3 = dict(
        dir_name = 'GCNNparameters',       # Directory name for results
        model_save = params.saveDirectory, # Path to save model checkpoints
        num_epochs = 30,                   # Number of epochs for training
        batch_size = 256,                  # Batch size furing training

        F = [32, 64],                      # Number of graph convolutional filters
        K = [20, 20],                      # Polynomial orders
        p = [2, 2],                        # Pooling sizes
        M = [512, 2],                      # Output dimensionality of fully connected layers

        regularization = 2e-5,             # L2 regularization
        learning_rate = 1e-4,              # Learning rate for training
        decay_rate = 0.95,                 # Decay rate
        decay_steps = int(0.1*30*Xtrain.shape[0] / 256),                    # Decays 10x with decay_rate
        imbalance = list(ytrain_added).count(0)/list(ytrain_added).count(1) # Likely imbalance
    )

    model3 = model.cgcnn(L, **params3)

    train_loss, val_loss, f1_scores, _, epochs, _, _, _, _ = model3.fit(Xtrain_added, ytrain_added, Xval, yval, False, True) # Finetune
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, train_loss, 'k.-', label='Training loss')
    ax1.plot(epochs, val_loss, 'k.--', label='Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(epochs, f1_scores, 'b.-', label='Validation F1-score')
    ax2.set_ylabel('F1-score', color='b')
    ax2.legend()
    plt.savefig(params.saveDirectory+'/finetuning.png')
    plt.show()

    prec_l, rec_l, f1_l, = [], [], []
    for i in range(len(Xtest)): # Test per subject
        _, f1, prec, rec, inter = model1.evaluate(Xtest[i], ytest[i], False)
        feat_test = np.vstack(inter)[:ytest[i].shape[0], :]
        inter_test1.append(feat_test)

        prec_l.append(prec)
        rec_l.append(rec)
        f1_l.append(f1)

    print("Finetuning \n Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                                  statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                                  statistics.mean(f1_l), statistics.stdev(f1_l)))

    with open(params.saveDirectory+'/fine_results.txt', 'w') as f:
        print("modelFine: \n Precision: {} +/- {} \n Recall: {} +/- {} \n F1-score: {} +/- {}".format(statistics.mean(prec_l), statistics.stdev(prec_l),
                                                                                                      statistics.mean(rec_l), statistics.stdev(rec_l),
                                                                                                      statistics.mean(f1_l), statistics.stdev(f1_l)), file=f)
        
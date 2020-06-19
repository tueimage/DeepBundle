"""
This code is adapted from the paper of Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst: 
Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, Neural Information Processing Systems (NIPS), 2016.

https://github.com/mdeff/cnn_graph
"""


import scipy.sparse
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os, collections, shutil
import tensorflow as tf


class base_model(object):

    def __init__(self):
        self.regularizers = []

    # High-level interface which runs the constructed computational graph.
    def predict(self, data, labels=None, sess=None):
        """
        Returns predictions, corresponding losses, and intermediate features
        Input:
        sess - The session in which the model has been trained
        data - Size N x M x 3
            N: number of signals (samples)
            M: number of vertices (each 3 features: x, y, z)
        labels - Size N
            N: number of signals (samples)
        sess - 
        """
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        intermediate_list = []
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], 3))
            tmp_data = data[begin:end, :, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin, :] = tmp_data
            feed_dict = {self.ph_data: batch_data}

            if labels is not None: # Compute loss
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]
            intermediate_list.append(sess.run(self.intermediate, feed_dict))

        if labels is not None:
            return predictions, loss * self.batch_size / size, intermediate_list
        return predictions, intermediate_list

    def evaluate(self, data, labels, mine, sess=None):
        """
        Runs one evaluation against the full epoch of data
        Returns the precision, recall, and F1 score
        Input:
        sess - The session in which the model has been trained
        data - Size N x M x 3
            N: number of signals (samples)
            M: number of vertices (each 3 features: x, y, z)
        labels - Size N
            N: number of signals (samples)
        mine - FP mining or not (boolean)
        """
        if mine: # FP mining
            FPidx = []
            predictions, loss_b, intermediate_list = self.predict(np.array(data), np.array(labels), sess)
            for i in range(labels.shape[0]):
                if labels[i] == 0 and predictions[i] == 1: # Definition of a FP
                    FPidx.append(i)
            return FPidx
        else:
            predictions, loss, intermediate_list = self.predict(data, labels, sess)
            precision, recall, F1_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
            return loss, F1_score, precision, recall, intermediate_list
        
    def fit(self, train_data, train_labels, val_data, val_labels, t_SNE, finetune, sess=None):
        """
        Trains the model and returns the resulting training losses.
        Input:
        train_data - Size N x M x 3
            N: number of signals (samples)
            M: number of vertices (each 3 features: x, y, z)
        train_labels: Size N
            N: number of signals (samples)
        val_data - List of arrays with size N x M x 3
            N: number of signals (samples)
            M: number of vertices (each 3 features: x, y, z)
        val_labels: List or arrays with size N
            N: number of signals (samples)
        t_SNE - Save intermediate features for t-SNE visualization or not (boolean)
        finetune - Finetune model (for FP mining) or not (boolean)
        """
        if finetune: # FP mining
            sess = self._get_session(sess) # Load parameters of best epoch of initial model
        else:
            sess = tf.Session(graph=self.graph)
            sess.run(self.op_init) # Initialize model

        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))        
        path = os.path.join(self._get_path('checkpoints'), 'model')
        
        train_loss, f1_scores, epochs, val_loss, intermediate_list_ep = [], [], [], [], []
        f1_max, prec_val_max, rec_val_max, epoch_tsne = 0, 0, 0, 0
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        eval_steps = [int(n*(num_steps/self.num_epochs)) for n in range(1, self.num_epochs+1)] # Per epoch
        
        for step in range(1, num_steps+1):
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0])) # Also adds random shuffle
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_labels = train_data[idx, :, :].astype('float16'), train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            epoch = step * self.batch_size / train_data.shape[0]
                        
            if step in eval_steps: # Per epoch
                train_loss.append(loss_average)
                val, f1, prec_val, rec_val, _ = self.evaluate(np.concatenate(val_data, axis=0), np.concatenate(val_labels, axis=0), False, sess) # Both validation subjects
                print('Step {} / {} (epoch {:.2f} / {}): loss {}, val {}, f1 {}'.format(step, num_steps, epoch, self.num_epochs, loss_average, val, f1))
                
                f1_scores.append(f1)
                val_loss.append(val)
                epochs.append(epoch)

                if f1 > f1_max: # Validation F1 score as performance metric
                    self.op_saver.save(sess, path, global_step=step) 
                    f1_max = f1
                    prec_val_max = prec_val
                    rec_val_max = rec_val
                    if t_SNE:
                        val_tsne, _, _, _, intermediate_list_ep = self.evaluate(val_data[0], val_labels[0], False, sess) # Only one validation subject
                        epoch_tsne = epoch
        
        _, _, _, _, intermediate_train = self.evaluate(train_data, train_labels, False, sess) # Get features from last model for SVM
        
        sess.close()
        
        return train_loss, val_loss, f1_scores, intermediate_list_ep, epochs, intermediate_train, epoch_tsne, prec_val_max, rec_val_max
   
    # Methods to construct the computational graph
    def build_graph(self, M_0):
        """
        Build the computational graph of the model.
        Input:
        M_0 - Number of vertices in original graph.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, 3), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                
            # Model.
            op_logits, intermediate = self.inference(self.ph_data)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization, self.imbalance) # Corresponding cross entropy loss
            self.op_train = self.training(self.op_loss, self.learning_rate, self.decay_steps, self.decay_rate) # Using SGD
            self.op_prediction = self.prediction(op_logits) 
            self.intermediate = tf.dtypes.cast(intermediate, tf.float16) 
            
            self.op_init = tf.global_variables_initializer() # Initialize variables            
            self.op_saver = tf.train.Saver(max_to_keep=1)
        
        self.graph.finalize()

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization, imbalance):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                if abs(imbalance - 1) < 1e-2: # Balanced (though rounding can lead to slight imbalance)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                else:
                    labels_oh = tf.one_hot(labels, 2)
                    labels_f = tf.to_float(labels_oh)
                    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=labels_f, logits=logits, pos_weight=imbalance) # Weighted variant
                    
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            
            loss = cross_entropy + regularization
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control') # The op return the learning rate
            return op_train

    # Helper methods
    def _get_path(self, folder):
        path = self.model_save 
        return os.path.join(path, folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given"""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        return var

class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation

    The following are hyper-parameters of graph convolutional layers
    They are lists, which length is equal to the number of gconv layers
        F: Number of features
        K: List of polynomial orders, i.e. filter sizes or number of hopes
        p: Pooling size
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level)
           Beware to have coarsened enough

    L: List of Graph Laplacians. Size M x M. One per coarsening level

    The following are hyper-parameters of fully connected layers
    They are lists, which length is equal to the number of fc layers
        M: Number of features per sample, i.e. number of hidden neurons
           The last layer is the softmax, i.e. M[-1] is the number of classes

    Training parameters:
        num_epochs:    Number of training epochs
        learning_rate: Initial learning rate
        decay_rate:    Base of exponential decay (no decay with 1)
        decay_steps:   Number of steps after which the learning rate decays
        imbalance:     Ratio of negative and positive labels in training set (balanced with 1)
        
    Regularization parameters:
        regularization: L2 regularizations of weights and biases
        batch_size:     Batch size. Must divide evenly into the dataset sizes

    Directories:
        dir_name:   Name for directories (summaries and model parameters)
        model_save: Path to save the modelparameters
    """
    def __init__(self, L, F, K, p, M, num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, regularization=0, batch_size=100, dir_name='', model_save='', imbalance=1):
        super().__init__()
        
        # Verify the consistency w.r.t. the number of layers
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)
        assert len(L) >= 1 + np.sum(p_log2)
        
        M_0 = L[0].shape[0] # Number of vertices in original graph
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L # Only use usefull Laplacians
        
        # Store attributes and bind operations
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps = decay_rate, decay_steps
        self.regularization = regularization
        self.batch_size = batch_size
        self.dir_name = dir_name
        self.model_save = model_save
        self.imbalance = imbalance
        
        # Build the computational graph
        self.build_graph(M_0)
    
    def rescale_L(self, L, lmax=2):
        """Rescale the Laplacian eigenvalues in [-1,1]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
        L /= lmax / 2
        L -= I
        return L  
    
    def chebyshev5(self, x, L, Fout, K):
        """Chebyshev spectralgraph convolution with polynomial order K"""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L
        L = scipy.sparse.csr_matrix(L)
        L = self.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def mpool(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def inference(self, x):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        """
        # Graph convolutional layers.
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.chebyshev5(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('relu'):
                    x = tf.nn.relu(x)
                with tf.name_scope('pooling'):
                    x = self.mpool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                if i == (len(self.M) - 2): # Second to last layer
                    intermediate = x 
        
        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x, intermediate
    
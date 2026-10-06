from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #对每一个测试数据，计算它和每一个训练数据的距离
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:     
                dists[i,j]=np.sqrt(np.sum((X[i]-self.X_train[j])**2,axis=0))                                                        #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO: 
            #dists[i]的形状(num_test,num_train)
            #X[i]的形状(3072,)
            #self.X_train的形状(num_train,3072) 
            #用广播机制，(X[i]-self.X_train)是二维数组，形状(num_train,3072)
            #测试数据和每个训练数据的距离dists[i,:],形状(num_train,)
            dists[i,:]=np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1))                                                          #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists. 
        # X.shape=(num_test,3072)
        # self.X_train.shape=(num_train,3072)
        # dist[i][j]是X第i行和self.X_train第j行之间的欧氏距离
        # 它的平方等于：
        # X这一整行元素平方和---X所有元素平方，压缩最后一维变成一维数组(num_test,)，末尾填充一维广播加到dists每一行
        # 加上self.X_train这一行所有元素平方和---self.X_train所有元素平方，每行求和变成(num_train,)，希望加到每一列，可以直接广播加上去
        # 减去两倍的X这一行和self.X_train这一行的点积---直接减去X和self.X_train的转置的矩阵乘法的两倍，即可
        dists=np.sqrt(np.expand_dims(np.sum(X**2,axis=1),axis=1)+np.sum((self.X_train)**2,axis=1)-2*(X@(self.X_train.T)))                                          #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:  
            indices=np.argsort(dists[i],axis=0,kind=None)[:k]
            closest_y=[self.y_train[idx] for idx in indices]                                                     #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################


            #########################################################################
            # TODO:    
            appear_times={}
            for item in closest_y:
              appear_times[item]=appear_times.get(item,0)+1
            max_times=max(appear_times.values())
            ans=None
            for item in closest_y:
              if appear_times[item]==max_times:
                if ans==None:
                  ans=item                               
                else:
                  if item<ans:
                    ans=item
            y_pred[i]=ans                    #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################


        return y_pred

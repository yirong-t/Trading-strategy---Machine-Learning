
import numpy as np


class RTLearner(object):


    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "ytang332"


    def study_group(self):
        return 'gburdell3'




    def random_feature(self, data_x):

        # np.random.seed(42)
        n = data_x.shape[1]
        feature_range = np.arange(0, n, 1)
        best_feature = np.random.choice(feature_range)
        return best_feature


    def build_tree(self, data_x, data_y):

        unique_val, uniques_cnt = np.unique(data_y, return_counts=True)
        if (data_x.shape[0] <= self.leaf_size) or len(uniques_cnt) == 1 :
            unique_val, uniques_cnt = np.unique(data_y, return_counts=True)
            prediction = unique_val[np.argmax(uniques_cnt)]
            return np.array([-1, prediction, np.nan, np.nan]).reshape(1,4)

        else:

            best_feature = self.random_feature(data_x)
            Split_val = np.median(data_x[:, best_feature])

            if data_x[:,best_feature].shape[0] == data_x[data_x[:,best_feature] <= Split_val].shape[0]:
                unique_val, uniques_cnt = np.unique(data_y, return_counts=True)
                prediction = unique_val[np.argmax(uniques_cnt)]
                return np.array([-1, prediction, np.nan, np.nan]).reshape(1,4)

            data_y = data_y.reshape(-1, 1)

            data = np.concatenate((data_x, data_y), axis=1)

            left = self.build_tree(data[data[:,best_feature] <= Split_val, :-1],
                                     data[data[:,best_feature] <= Split_val, -1].reshape(-1,1))

            right = self.build_tree(data[data[:,best_feature] > Split_val, :-1],
                                      data[data[:,best_feature] > Split_val, -1].reshape(-1,1))

            root = np.array([int(best_feature), Split_val, 1, left.shape[0] + 1])

            tree = np.vstack([root, left,right])


            return np.vstack([root, left,right])


    def add_evidence(self, data_x, data_y):

        data_y = data_y.reshape(-1, 1)
        self.tree = self.build_tree(data_x, data_y)

        return self.tree



    def query(self, X_test):

        Y_test = np.array([])


        for i in range(X_test.shape[0]):
            row_num = 0
            while row_num < self.tree.shape[0]:

                if (int(self.tree[row_num,0]) == -1):
                    Y_test = np.append(Y_test, self.tree[row_num, 1])
                    row_num +=  self.tree.shape[0]

                elif X_test[i,int(self.tree[row_num,0])] <= self.tree[row_num,1]:
                    row_num += 1

                elif X_test[i,int(self.tree[row_num,0])] > self.tree[row_num,1]:
                    row_num += int(self.tree[row_num, 3])

                else:
                    print("error")


        return Y_test

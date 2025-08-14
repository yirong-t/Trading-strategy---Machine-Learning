
import numpy as np


class BagLearner(object):

    def __init__(self,learner, kwargs={}, bags=30,boost=False, verbose=False):

        self.bags=bags
        self.boost=boost
        self.verbose=verbose
        self.extra_param = kwargs.pop
        self.kwargs=kwargs
        self.models = []
        self.learners = [learner(**self.kwargs) for _ in range(bags)]
        """
        Constructor method
        """

    def author(self):
        return "ytang332"


    def study_group(self):
        return 'gburdell3'


    def generate_dataset(self,data_x,data_y):
        data_y = data_y.reshape(-1,1)
        data = np.concatenate((data_x, data_y),axis=1)
        size = int(len(data_x))
        index = np.random.choice(data.shape[0], size=size, replace=True)
        random_data = data[index]
        X_train, Y_train = random_data[:size,:-1], random_data[:size,-1].reshape(-1,1)


        return X_train,Y_train


    def bag_learner(self,data_x,data_y):

        for learner in self.learners:
            X_train, Y_train = self.generate_dataset(data_x, data_y)
            learner.add_evidence(X_train, Y_train)



    def add_evidence(self, data_x, data_y):

        data_y = data_y.reshape(-1,1)
        self.bag_learner(data_x, data_y)


    def query(self, X_test):

        Y_pred = np.concatenate([learner.query(X_test).reshape(-1, 1) for learner in self.learners], axis=1)
        Y_prediction = np.zeros(X_test.shape[0])


        for i in range(X_test.shape[0]):
            unique, counts = np.unique(Y_pred[i], return_counts=True)
            # print(unique, counts)
            Y_prediction[i] = unique[np.argmax(counts)]


        return Y_prediction



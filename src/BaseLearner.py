import copy

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from src import Parser


class BaseLearner():
    def __init__(self, base, seeds_indices, X, y):
        self.X = X
        self.y = y
        self.train_indices = [s for s in seeds_indices]
        self.base = base
        self.unobserved_indices = []
        for i in range(len(X)):
            if i not in self.train_indices:
                self.unobserved_indices.append(i)

    def predict_acc(self):
        X_train = [self.X[i] for i in self.train_indices]
        y_train = [self.y[i] for i in self.train_indices]
        X_test = [self.X[i] for i in self.unobserved_indices]
        y_test = [self.y[i] for i in self.unobserved_indices]

        total_error = 0
        self.base.fit(X_train, y_train)
        return self.base.score(X_test, y_test)

    def cross_validation(self, fold):
        kf = KFold(n_splits=fold)
        cv_acc = 0
        for train, test in kf.split(self.train_indices):
            train_X = [self.X[self.train_indices[i]] for i in train]
            train_y = [self.y[self.train_indices[i]] for i in train]
            test_X = [self.X[self.train_indices[i]] for i in test]
            test_y = [self.y[self.train_indices[i]] for i in test]
            self.base.fit(train_X, train_y)
            cv_acc += self.base.score(test_X, test_y)
        return cv_acc / fold

    def increment_train(self):
        choice = np.random.choice(self.unobserved_indices)
        self.unobserved_indices.remove(choice)
        self.train_indices.append(choice)

    def train_size(self):
        return len(self.train_indices)



if __name__ == "__main__":
    X_,y_,labels = Parser.parse_csv("../data/02450ProjectExpressionData.csv", 20)
    rf = RandomForestClassifier()
    bl = BaseLearner(rf,[1,2,3,4,5],X_,y_)
    print(bl.predict_acc())
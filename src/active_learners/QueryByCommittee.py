from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import Parallel, delayed

from src import Parser
from src.BaseLearner import BaseLearner
import numpy as np
import collections


# default dimension 20 committee members
from src.active_learners.UncertaintySampling import UncertaintySampling


def vote_entropy(arr):
    committee = 20
    count = collections.Counter(arr)
    entropy = 0
    for v in count.values():
        entropy += v/committee*np.log(v/committee)
    return -1*entropy


class QueryByCommittee(BaseLearner):
    def increment_train(self):
        committee = 20
        size = self.X.shape[1]
        votes = np.zeros((len(self.unobserved_indices), 0))
        #for n in range(committee):
        def find_vote(n):
            X_train = [[self.X[i][n]] for i in self.train_indices]
            y_train = [self.y[i] for i in self.train_indices]
            X_test = [[self.X[i][n]] for i in self.unobserved_indices]
            self.base.fit(X_train, y_train)
            y_predict = [[v] for v in self.base.predict(X_test)]
            return y_predict
        votes = Parallel(n_jobs=16)(delayed(find_vote)(n)for n in range(committee))
        #print(votes)
        votes = np.concatenate(votes,axis=1)
        entropy = map(vote_entropy, votes)
        index = max(enumerate(entropy), key=lambda x: x[1])[0]
        choice = self.unobserved_indices.pop(index)
        self.train_indices.append(choice)


if __name__ == "__main__":
    X_, y_,labels = Parser.parse_csv("../../data/02450ProjectExpressionData.csv", 20)
    rf = RandomForestClassifier()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    bl = QueryByCommittee(clf, [0,1, 2, 3, 4, 5,6,7,8,9,10,20,50], X_, y_)
    print(bl.train_indices)
    print(bl.predict_acc())

    bl.increment_train()
    print(bl.train_indices)
    print(bl.predict_acc())

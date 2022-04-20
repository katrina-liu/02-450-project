import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import Parser
from src.BaseLearner import BaseLearner


class UncertaintySampling(BaseLearner):
    def increment_train(self):
        X_train = [self.X[i] for i in self.train_indices]
        y_train = [self.y[i] for i in self.train_indices]
        X_test = [self.X[i] for i in self.unobserved_indices]
        self.base.fit(X_train, y_train)
        test_proba = self.base.predict_proba(X_test)
        confidence = map(max, test_proba)

        min_index = min(enumerate(confidence), key=lambda x: x[1])[0]
        choice = self.unobserved_indices.pop(min_index)
        self.train_indices.append(choice)


if __name__ == "__main__":
    X_, y_,labels = Parser.parse_csv("../../data/02450ProjectExpressionData.csv", 20)
    rf = RandomForestClassifier()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    bl = UncertaintySampling(clf, [0,1, 2, 3, 4, 5,6,7,8,9,10,20,50], X_, y_)
    print(bl.train_indices)
    print(bl.predict_acc())

    bl.increment_train()
    print(bl.train_indices)
    print(bl.predict_acc())

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import Parser
from src.BaseLearner import BaseLearner
from src.active_learners.UncertaintySampling import UncertaintySampling


class ErrorReduction(BaseLearner):
    def increment_train(self):
        X_train = [self.X[i] for i in self.train_indices]
        y_train = [self.y[i] for i in self.train_indices]
        X_test = [self.X[i] for i in self.unobserved_indices]
        test_proba = self.base.predict_proba(X_test)
        loss = []
        for i in range(len(self.unobserved_indices)):
            loss_i = 0
            x_i = X_test[i]
            proba = test_proba[i]
            new_test = []
            for j in range(len(self.unobserved_indices)):
                if j != i:
                    new_test.append(X_test[j])
            new_x_train = X_train + [x_i]
            for j in range(len(proba)):
                p_i = proba[j]
                y_i = self.base.classes_[j]
                new_y_train = y_train + [y_i]
                self.base.fit(new_x_train, new_y_train)
                probas = self.base.predict_proba(new_test)
                loss_i += p_i * (sum(map(lambda x: 1 - max(x), probas)))
            loss.append((i, loss_i))
            print(loss)
        min_index = min(loss, key=lambda x: x[1])[0]
        self.train_indices.append(self.unobserved_indices.pop(min_index))


if __name__ == "__main__":
    X_, y_,labels = Parser.parse_csv("../../data/02450ProjectExpressionData.csv", 20)
    rf = RandomForestClassifier()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    bl = ErrorReduction(clf, [0,1, 2, 3, 4, 5,6,7,8,9,10,20,50], X_, y_)
    print(bl.train_indices)
    print(bl.predict_acc())

    bl.increment_train()
    print(bl.train_indices)
    print(bl.predict_acc())
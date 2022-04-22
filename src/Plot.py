import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import Parser
from src.BaseLearner import BaseLearner
from src.active_learners.ErrorReduction import ErrorReduction
from src.active_learners.QueryByCommittee import QueryByCommittee
from src.active_learners.UncertaintySampling import UncertaintySampling
import matplotlib.pyplot as plt

X_, y_, labels = Parser.parse_csv("../data/Adrenal.csv", 20)
rf = RandomForestClassifier()
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))

stop = len(X_)//2
seed_num = len(X_) // 10
pre_acc = [[],[],[],[]]
train_size = []
for _ in range(10):
    seeds = np.random.permutation(len(X_))[:seed_num]
    print(seeds)
    rf_learners = [BaseLearner(rf, seeds, X_, y_),
                   UncertaintySampling(rf, seeds, X_, y_),
                   QueryByCommittee(rf, seeds, X_, y_),
                   ErrorReduction(rf, seeds, X_, y_)]

    predict_accuracy = [[],[],[],[]]
    for i in range(4):
        train_size = []
        learner = rf_learners[i]
        #print(learner.train_size())
        while learner.train_size() <= stop:
            train_size.append(learner.train_size())
            predict_accuracy[i].append(learner.predict_acc())
            learner.increment_train()
        pre_acc[i].append(predict_accuracy[i])
print(train_size)
print(pre_acc)
print(pre_acc)
print(len(train_size),np.mean(pre_acc[0], axis=0))

plt.figure()
plt.errorbar(train_size,
                 np.mean(pre_acc[0], axis=0),
                 yerr=np.std(pre_acc[0], axis=0),
                 label="Random", fmt='-o', capsize=3)

plt.errorbar(train_size,
                 np.mean(pre_acc[1], axis=0),
                 yerr=np.std(pre_acc[1], axis=0),
                 label="Uncertainty Sampling", fmt='-o', capsize=3)
plt.errorbar(train_size,
                 np.mean(pre_acc[2], axis=0),
                 yerr=np.std(pre_acc[2], axis=0),
                 label="Query By Committee", fmt='-o', capsize=3)

plt.errorbar(train_size,
                 np.mean(pre_acc[3], axis=0),
                 yerr=np.std(pre_acc[3], axis=0),
                 label="Expected Error Reduction", fmt='-o', capsize=3)
plt.legend()
plt.xlabel("Train size")
plt.ylabel("Accuracy")
plt.title("Random Forest - Adrenal Cancer")
plt.savefig("../figures/rd_adrenal.png")





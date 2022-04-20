import csv

import numpy as np


def parse_csv(filename, x_dim):
    """
    Parse the data set from csv file
    :param filename: source file name
    :param x_dim: dimension of the x data
    :return: x data and y data
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader,None)
        X = []
        y = []
        labels = []
        for row in reader:
            #print(row)
            X.append([float(elem) for elem in row[:x_dim]])
            if row[x_dim] not in labels:
                labels.append(row[x_dim])
            y.append(labels.index(row[x_dim]))
        return np.array(X), np.array(y), labels


if __name__ == "__main__":
    X_, y_, labels_ = parse_csv("../data/02450ProjectExpressionData.csv", 20)
    print(X_, y_, labels_)

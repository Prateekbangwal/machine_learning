import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from machine_learning.perceptron_classifier import Perceptron
from matplotlib.colors import ListedColormap


class IrisBinaryClassification():
    """
    labels = iris- setosa and iris - verisicolor
        features = first column of dataset --> sepal length
        and third feature column of dataset --> petal length
        """
    def __init__(self):
        self.a = 1
        self.perceptron = Perceptron(eta=0.1, n_iter=10)

    def perceptron_classification_on_iris_dataset(self):
        """select setosa and versicolor as labels"""
        y = df.iloc[0:100, 4].values
        y = np.where(y == "Iris-setosa", -1, 1)
        """extracts features sepal length and petal length"""
        X = df.iloc[0:100, [0, 2]].values
        plt.scatter(X[:50,0],X[:50,1],color = 'red', marker = 'o', label = 'setosa')
        plt.scatter(X[50:100, 0], X[50:100,1], color = 'blue', marker = 'x', label = 'versivcolor')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc= 'best')
        plt.show()
        ppn = self.perceptron.fit(X,y)
        print(len(ppn.errors_))
        plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker = 'o')
        plt.xlabel("epochs")
        plt.ylabel("Number of updates")
        plt.show()
        self.plot_decision_regions(X,y, classifier=ppn)

    def plot_decision_regions(self, X, y , classifier, resolution = 0.02):
        """setup marker generator and color map"""
        markers  = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        """plot the decision surface"""
        x1_min, x1_max = X[:,0].min() -1, X[:, 0].max() +1
        x2_min, x2_max = X[:,1].min() -1, X[:,1].max() +1
        xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution), np.arange(x2_min, x2_max, resolution))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contour(xx1,xx2, Z, alpha = 0.3, cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        """plot the class examples"""
        for idx, c1 in enumerate(np.unique(y)):
            plt.scatter(x = X[y==c1, 0],y = X[y ==c1, 1], alpha=0.8,c =colors[idx], marker = markers[idx], label = c1, edgecolors='black')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc='best')
        plt.show()




    def execute(self):
        """program entrance function"""
        self.perceptron_classification_on_iris_dataset()



if __name__ == '__main__':
    obj = IrisBinaryClassification()
    s = os.path.join('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    df = pd.read_csv(s, header=None, encoding='utf-8')
    print(df.tail())
    obj.execute()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
#from sklearn.linear_model import Perceptron
from Perceptron import Perceptron
from SVMDual import SVMDual
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.data', header=None)

# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ppn = Perceptron(epochs=10, tau=0.01)

ppn.train(X_train, y_train)

print('Weights: %s' % ppn.w_)

plot_decision_regions(X_test, y_test, clf=ppn)
plt.title('Perceptron on setosa and versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

print('Total number of misclassification: %d of 100' % (y_test != ppn.predict(X_test)).sum())

"""
Problems with Perceptrons
Although the perceptron classified the two Iris flower classes perfectly,
convergence is one of the biggest problems of the perceptron.
Frank Rosenblatt proofed mathematically that the perceptron learning rule converges
if the two classes can be seperated by linear hyperplane,
but problems arise if the classes cannot be seperated perfectly by a linear classifier.
To demonstrate this issue, we wll use 2 different classes and features from Iris dataset.
"""

# versicolor and virginica
y2 = df.iloc[50:150, 4].values
y2 = np.where(y2 == 'Iris-virginica', -1, 1)

# sepal width and petal width
X2 = df.iloc[50:150, [1, 3]].values
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

ppn = Perceptron(epochs=10, tau=0.01)
ppn.train(X_train, y_train)

plot_decision_regions(X_test, y_test, clf=ppn)
plt.title('Perceptron on versicolor and virginica')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()
#
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()
print('Total number of misclassification: %d of 100' % (y_test != ppn.predict(X_test)).sum())

svm = SVMDual(max_iter=100, kernel_type='linear', C=1.0, gamma=1.0)
svm.fit(X_train, y_train)
plot_decision_regions(X_test, y_test, clf=svm)
plt.title('SVM on versicolor and virginica')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()
print('Total number of misclassification: %d of 100' % (y_test != svm.predict(X_test)).sum())
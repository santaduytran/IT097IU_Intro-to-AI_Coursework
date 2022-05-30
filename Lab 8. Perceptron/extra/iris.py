import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from Perceptron import Perceptron
from SVMDual import SVMDual
from sklearn.model_selection import train_test_split
# Import packages to visualize the classifer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

# Import packages to do the classifying
import numpy as np
from sklearn.svm import SVC

def plot_decision_regions_(X, y, classifier, test_idx=None, resolution=0.02):
    #for i in range(0, len(X[0])-1):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(x1_min, x1_max, x2_min, x2_max)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

    plt.title('Test')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def transform(X, y):
    for i in range(0, len(X) - 1):
        for j in range(i+1, len(X)):
            if (X[i] == X[j]).all() and (y[i] != y[j]).all():
                X[j] = -(X[i])
    return X, y

df = pd.read_csv('iris.data', header=None)

# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#plot_decision_regions(X_train, y_train, None)

ppn = Perceptron(epochs=10, tau=0.01)
ppn.train(X_train, y_train)
print('Weights: %s' % ppn.w_)
plot_decision_regions(X_test, y_test, ppn)


plt.title('Perceptron on setosa and versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

print('Total number of perceptron misclassifications: %d of 100' % (y_test != ppn.predict(X_test)).sum())

"""
Problems with Perceptrons
Although the perceptron classified the two Iris flower classes perfectly, 
convergence is one of the biggest problems of the perceptron. 
Frank Rosenblatt proofed mathematically that the perceptron learning rule converges 
if the two classes can be separated by linear hyperplane,
but problems arise if the classes cannot be separated perfectly by a linear classifier. 
To demonstrate this issue, we will use two different classes and features from the Iris dataset.
"""

#versicolor and virginica
y2 = df.iloc[50:150, 4].values
y2 = np.where(y2 == 'Iris-virginica', -1, 1)

# sepal width and petal width
X2 = df.iloc[50:150, [1, 3]].values

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

ppn = Perceptron(epochs=10, tau=0.01)
ppn.train(X_train, y_train)

plot_decision_regions(X_test, y_test, ppn)
plt.title('Perceptron on versicolor and virginica')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()
#
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()
print('Total number of perceptron misclassifications: %d of 100' % (y_test != ppn.predict(X_test)).sum())

#C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40, 50]
#g = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40, 50]
#for i in range(len(C)):

# ppn = PerceptronDual(epochs=1000, tau=0.01, kernel_type='quadratic', C = 10, gamma = 1)
# ppn.train(X_train, y_train)
#
# plot_decision_regions(X_test, y_test, ppn)
# plt.title('Perceptron on versicolor and virginica')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()
# #
# plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Misclassifications')
# plt.show()
# print('Total number of perceptron-dual misclassifications: %d of 100' % (y_test != ppn.predict(X_test)).sum())

def transform(X):
    X_ = []
    for row in range(0, X.shape[0]):
        X_.append(np.linspace(X[row][0], X[row][1], 4).tolist())
    X_ = np.asarray(X_)
    print('Hello: ', X_)
    return X_

#X_train = transform(X_train)
#X_test = transform(X_test)

svm = SVMDual(max_iter=1000, kernel_type='quadratic', C=10, epsilon=1e-10, gamma=1)
svm.fit(X_train, y_train)
plot_decision_regions(X_test, y_test, svm)
print('Total number of misclassifications using svm with quadratic kernel on testing data : %d of 100' % (y_test != svm.predict(X_test)).sum())
plt.title('Test SVM with quadratic kernel on versicolor and virginica')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()


svm = SVMDual(max_iter=1000, kernel_type='rbf', C=10, epsilon=1e-10, gamma=1)
svm.fit(X_train, y_train)
plot_decision_regions(X_test, y_test, svm)
print('Total number of misclassifications using svm with rbf kernel on testing: %d of 100' % (y_test != svm.predict(X_test)).sum())
plt.title('Test SVM with RBF kernel on versicolor and virginica')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

# ada = AdalineSGD(epochs=10, eta=0.01).train(X2, y2)
# plt.plot(range(1, len(ada.cost_)+1), np.log10(ada.cost_), marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('log(Sum-squared-error)')
# plt.title('Adaline - Learning rate 0.01')
# plt.show()
#
# ada = AdalineSGD(epochs=10, eta=0.0001).train(X2, y2)
# plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Sum-squared-error')
# plt.title('Adaline - Learning rate 0.0001')
# plt.show()
#
# # standardize features
# X_std = np.copy(X2)
# X_std[:,0] = (X2[:,0] - X2[:,0].mean()) / X2[:,0].std()
# X_std[:,1] = (X2[:,1] - X2[:,1].mean()) / X2[:,1].std()
# ada = AdalineSGD(epochs=15, eta=0.01)
#
# ada.train(X_std, y2)
# plot_decision_regions(X_std, y, clf=ada)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.show()
#
# plt.plot(range(1, len( ada.cost_)+1), ada.cost_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Sum-squared-error')
# plt.show()
#
# print('Total number of misclassifications: %d of 100' % (y2 != ada.predict(X2)).sum())

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def plot_cm(cm, labels=("Normal", "Heart disease"), normalize=False):
    """

    :param cm:
        Not normalized confusion matrix
    :return:
    """
    df_cm = pd.DataFrame(cm / np.sum(cm) if normalize else cm, index=labels, columns=labels)  # normalized
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.show()


def plot_prediction(clf, X_train, X_test, y_train):
    fig, ax = plt.subplots(1, figsize=(11, 7))
    # plotSvm(xTrain3, yTrain3, support=model30.supportVectors, label='Training', ax=ax)
    ax.set_title('Decision surface of SVM')
    ax.set_ylabel('feature x1')
    ax.set_xlabel('feature x2')
    ax.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, label='train data')  # train data
    ax.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=clf.predict(X_test), label='test data')  # test data scatter
    ax.legend()

    # Estimate and plot decision boundary
    xx = np.linspace(np.min(X_train), np.max(X_train), 50)
    X0, X1 = np.meshgrid(xx, xx)
    xy = np.vstack([X0.ravel(), X1.ravel()]).T

    Y = clf.predict(xy).reshape(X0.shape)

    ax.contour(X0, X1, Y, colors='k', levels=[-1, 0], alpha=0.3, linestyles=['-.', '-'])

    plt.show()


def accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)


def recall(cm):
    return np.diag(cm) / np.sum(cm, axis=1)


def precision(cm):
    return np.diag(cm) / np.sum(cm, axis=0)


def f1_score(cm):
    pass


def roc(cm):
    pass


def auc(cm):
    pass


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance
        # row  = 1 sample, columns = feature >> we have to transform because of np.cov()
        cov = np.cov(X.T)
        # eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #  v[:, i]
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0: self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

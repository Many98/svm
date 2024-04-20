import numpy as np
from numba import jit
from numba_progress import ProgressBar
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, pairwise_distances
from sklearn.svm import SVC
from tqdm import tqdm
import matplotlib.pyplot as plt

@jit(nopython=True)  # just in time compilation
def loss_(alpha, b, gram, y, C):
    sum = 0.0
    for i in range(gram.shape[0]):
        if y[i] * (gram[i] @ alpha.T + b) < 1:
            sum += 1.0 - y[i] * (gram[i] @ alpha.T + b)
    return C * sum + 0.5 * alpha @ gram @ alpha.T


class SVM(object):
    """
    Sample implementation of support vector machine using kernel trick.
    Only for binary classification.
    Ref.: https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf ; page 420 -- for primal and dual
          problem definition and quadratic programming solution
          https://stats.stackexchange.com/questions/215524/is-gradient-descent-possible-for-kernelized-svms-if-so-why-do-people-use-quadr
           -- kernel reformulation using hinge loss; which will be used in implementation
    """
    def __init__(self, kernel, C, num_classes=None, degree=2.0, gamma=None, coef=0.0, eps=1e-2,
                 use_sklearn_reference=False):
        """

        :param kernel: str
            Specifies kernel function used in svm.
            Can be one of `linear`, `poly`, `rbf`
        :param C: float
            Cost parameter. Serves as regularization and as upper limit to total number of miss-classification enabled.
        :param num_classes: int
            Number of classes. (SVM is natively used for binary classification).
            `One vs one` approach will be used if > 2. For n classes it results in combination(n, 2)=n(n-1) / 2
            classifiers. To speed up process parallel processing can be set
        :param use_sklearn_reference: bool
            Whether to use scikit-learn implementation instead.
        """
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.C = C
        self.eps = eps
        self.num_classes = num_classes
        self.use_sklearn = use_sklearn_reference
        self.loss = []
        self.alphas = []
        self.bs = []
        self.alpha = None
        self.b = None
        if use_sklearn_reference:
            self.svm = SVC(kernel=kernel, C=C, degree=degree, gamma='auto' if gamma is None else gamma,
                           coef0=coef, tol=eps)

    def fit(self, X, y, lr=0.00001, n_iters=1000):
        """
        :param lr: float
            learning rate
        :param n_iters: int
            number of iterations of gradient descent
        :param X: np.array
            2d design matrix of shape <num_samples x num_features>
        :param y: np.aray
            2d matrix of shape <num_samples x num_classes>
        :return:
        """
        if self.use_sklearn or len(np.unique(y)) > 2:
            self.svm.fit(X, y)
            if len(np.unique(y)) > 2:
                print('Using scikit-learn SVM because number of classes exceeded 2.')
        else:

            self.alpha = np.zeros(X.shape[0])  # row vector
            self.b = 0
            gram_matrix = self._gram_matrix(X, X)  # gram (kernel) matrix is symmetric

            with ProgressBar(total=n_iters) as progress:
                self.alphas, self.bs, self.loss = SVM._gradient_descent(self.alpha, self.b,
                                                                        gram_matrix, y, lr,
                                                                        self.C, n_iters,
                                                                        self.eps, progress)
            self.loss = np.where(self.loss == 0.0, np.max(self.loss), self.loss)
            best_iter = np.argmin(self.loss)
            self.alpha = self.alphas[best_iter]
            self.b = self.bs[best_iter]

            # we need to store support vectors from design matrix (for prediction)
            support_indices = np.abs(self.alpha) > 1e-8  # alpha less then this threshold are considered zero
            self.support_vectors = X[support_indices]
            self.support_alpha = self.alpha[support_indices]

    @staticmethod
    @jit(nopython=True, nogil=True)   # much faaaaaaster
    def _gradient_descent(alpha, b, gram_matrix, y, lr, C, n_iters, eps, progress_proxy):
        alphas = np.zeros((n_iters, alpha.shape[0]))
        bs = np.zeros((n_iters, 1))
        loss = np.zeros((n_iters, 1))

        for i in range(n_iters):
            loss[i] = loss_(alpha, b, gram_matrix, y, C)
            alphas[i] = alpha
            bs[i] = b
            for idx in range(gram_matrix.shape[0]):
                if y[idx] * (gram_matrix[idx] @ alpha.T + b) >= 1:
                    alpha -= lr * (gram_matrix @ alpha.T)
                else:
                    alpha -= lr * ((gram_matrix @ alpha.T) - C * gram_matrix[idx] * y[idx])
                    b += lr * y[idx] * C
            if np.abs(loss[i] - loss_(alpha, b, gram_matrix, y, C)) < eps:
                loss = loss[0:i+1]
                alphas = alphas[0:i+1]
                bs = bs[0:i+1]
                break
            progress_proxy.update(1)
        return alphas, bs, loss

    def _one_vs_one(self):
        pass
        # TODO implement me

    def _one_vs_rest(self):
        pass
        # TODO implement me

    def _plot_loss(self):
        fig, ax = plt.subplots()
        plt.plot([i for i in range(len(self.loss))], self.loss)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss profile')
        plt.show()

    def predict(self, X):
        """

        :param X: np.ndarray
            Design matrix with vectors for prediction / evaluation
        :return:
            predicted value {-1, 1}
        """
        if self.use_sklearn:
            return self.svm.predict(X)

        gram = self._gram_matrix(X, self.support_vectors)
        return np.sign(self.support_alpha @ gram.T + self.b)

    def _gram_matrix(self, X, Y=None):
        """
        Method to get gram (kernel) matrix
        :param X: np.ndarray
            design matrix of shape <num_samples, num_features>
        :return: np.ndarray
            gram matrix of shape <num_samples, num_samples>
        """
        if Y is None:
            Y = X
        if self.gamma is None:
            self.gamma = 1 / X.shape[1]
        if self.kernel == 'linear':
            return X @ Y.T * self.gamma
        elif self.kernel == 'rbf':
            pairwise_dists = cdist(X, Y, 'euclidean')
            return np.exp(- self.gamma * (pairwise_dists ** 2))
        elif self.kernel == 'poly':
            return ((X @ Y.T) * self.gamma + self.coef) ** self.degree
        else:
            raise Exception('Not supported kernel.')

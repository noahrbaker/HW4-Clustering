import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if type(k) != int:
            raise TypeError("k arg must be type int")
        if type(tol) != float:
            raise TypeError("tol arg must be type float")
        if any(input < 0 for input in [k, tol]):
            raise ValueError("k and tol both must be >0")
        if type(max_iter) != int:
            raise TypeError("max_iter arg must be type int")
        if max_iter < 1:
            raise ValueError("max_iter must be >=1")

        # passes the above, assume they are ok
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = None
        self.mse = None
        

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        
        if not isinstance(mat, np.ndarray):
            raise ValueError("mat must be a numpy array")
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D")

        # starting centroids
        mat_indices = np.random.choice(range(mat.shape[0]), self.k, replace=False)
        self.centroids = mat[mat_indices]

        # go through to the max iterations
        for i in range(self.max_iter):
            dist = cdist(mat, self.centroids, metric='euclidean')
            closest = np.argmin(dist, axis=1)
            
            new_centroids = np.array([np.mean(mat[closest == c], axis=0) for c in range(self.k)])

            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break

            self.centroids = new_centroids
        
        # record the error
        self.mse = np.mean(np.min(cdist(mat, self.centroids), axis=1) ** 2)


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError("mat must be a numpy array")
        if mat.ndim !=2:
            raise ValueError("mat must be 2D")
        if self.centroids is None:
            raise ValueError("fit method must be run before the predict method")

        dist = cdist(mat, self.centroids)
        labels = np.argmin(dist, axis=1)
        return labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        if self.mse is None:
            raise ValueError("fit method must be run before the get_error method")

        return self.mse



    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        if self.centroids is None:
            raise ValueError("fit method must be run before the get_centroids method")
        
        return self.centroids
import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            raise ValueError("X and y both have to be numpy arrays")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X has to be 2D and y has to be 1D")

        silly_score = np.zeros(X.shape[0])  # initialize the silhouette array
        
        k = np.unique(y)    # get the unique labels

        # if we just have one cluster, return 0 for the score by default
        if len(k) == 1:
            return silly_score

        dist = cdist(X, X)

        for i in range(y.shape[0]):
            y_ind = y[i]    # current label
            x_row = dist[i] # just the row
            x_in_y = x_row[y == y_ind]  # those in the row with the label
            
            # check to be sure the clusters are not one point
            if x_in_y[x_in_y != 0].size == 0:   
                silly_score[i] = 0
                continue
            
            # how far are points in the same cluster on average?
            a_part = np.mean(x_in_y[x_in_y != 0])

            # smallest mean distance for each label not assigned
            clust_dists = [np.mean(x_row[y == sub_index]) for sub_index in k if sub_index != y_ind]
            b_part = np.min(clust_dists) if clust_dists else 0

            silly_score[i] = (b_part - a_part)/(np.max([a_part, b_part]))
        
        return silly_score
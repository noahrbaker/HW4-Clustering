"""
BMI 203: Biocomputing Algorithms - Winter 2025
Noah Baker
HW 4: Clustering
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = '0.0.2'
# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster import utils


x_test, lab_test = utils.make_clusters(n=500, m=2, k=3, seed=42)


def test_kmeans():
    with pytest.raises(ValueError):
        kmeans = KMeans(k=-1)
    with pytest.raises(TypeError):
        kmeans = KMeans(k=3, tol=int(1))
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3, tol=-1e-6)
    with pytest.raises(TypeError):
        kmeans = KMeans(k=3, tol=1e-6, max_iter=1.7)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3, tol=1e-6, max_iter=-1)


def test_kmeans_fit():
    kmeans = KMeans(k=3)
    assert kmeans.centroids is None, "kmeans object is too young to have centroids"
    assert kmeans.mse is None, "kmeans object should not have an MSE yet"
    kmeans.fit(x_test)
    assert kmeans.centroids.shape == (3, x_test.shape[1]), "kmeans object has the wrong centroids"
    assert isinstance(kmeans.mse, float), "kmeans object should have an MSE after fit method"

    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.fit(5)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.fit(np.array([5]))


def test_kmeans_predict():
    kmeans = KMeans(k=3)
    kmeans.fit(x_test)
    labels = kmeans.predict(x_test)
    assert set(labels) == set(lab_test), "The labels are all of the same numerical range"
    assert labels.shape[0] == x_test.shape[0], "Not every input was given a label"
    assert all(label in set(lab_test) for label in labels), "Some labels are not within the known set of labels"

    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.fit(x_test)
        kmeans.predict(5)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.fit(x_test)
        kmeans.predict(np.array([5]))


def test_kmeans_getcentroids():
    kmeans = KMeans(k=3)
    assert kmeans.centroids is None, "kmeans object is too young to have centroids"
    kmeans.fit(x_test)
    assert kmeans.centroids.shape == (3, x_test.shape[1]), "kmeans object has the wrong centroids"
    assert kmeans.get_centroids() == kmeans.centroids, "kmeans not returning the same centroids"

    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.get_centroids()


def test_kmeans_geterror():
    kmeans = KMeans(k=3)
    assert kmeans.mse is None, "kmeans object should not have an MSE yet"
    kmeans.fit(x_test)
    assert isinstance(kmeans.mse, float), "kmeans object should have an MSE after fit method"
    assert kmeans.get_error() == kmeans.mse, "kmeans not returning the same MSE"

    with pytest.raises(ValueError):
        kmeans = KMeans(k=3)
        kmeans.get_error()
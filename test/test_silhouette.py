# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster import utils
from sklearn.metrics import silhouette_samples


x_test, lab_test = utils.make_clusters(n=500, m=2, k=3, seed=42)
kmeans = KMeans(k=3)
kmeans.fit(x_test)
labels = kmeans.predict(x_test)

def test_silhouette():
    try:
        silhouette = Silhouette()
        assert True
    except:
        assert False, "Silhouette class failed to load"


def test_silhouette_score():
    silhouette = Silhouette()
    with pytest.raises(ValueError):
        silhouette.score(X=5, y=5)
    with pytest.raises(ValueError):
        silhouette.score(X=x_test, y=np.array([[1, 2], [1, 2]]))
    with pytest.raises(ValueError):
        silhouette.score(X=np.array([1]), y=labels)

    scores = silhouette.score(x_test, labels)
    assert scores.shape[0] == x_test.shape[0], "Not every point got a silhouette score"
    assert all(scores >= -1) and all(scores <= 1), "silhouette scores are out of bounds [-1, 1]"

    # test against sklearn
    sk_scores = silhouette_samples(x_test, labels)
    assert np.all(np.isclose(scores, sk_scores)), "sklearn silhouette scores are different than custom method"
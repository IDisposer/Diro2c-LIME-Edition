import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor


class USLime:
    def __init__(
        self,
        n_neighbors=50,
        n_samples=500,
        noise_scale=0.01,
        random_state=None,
    ):
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.random_state = random_state

        self.X_train = None
        self.delta_fn = None
        self.nn = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, delta_fn):
        """
        Parameters
        ----------
        X : np.ndarray
            Training data used for neighborhood sampling
        delta_fn : callable
            Function: delta_fn(X) -> shape (n_samples,)
            Returns difference value for each sample
        """
        self.X_train = X
        self.delta_fn = delta_fn

        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn.fit(X)

        return self

    def _sample_locally(self, x):
        distances, indices = self.nn.kneighbors([x])
        neighbors = self.X_train[indices[0]]

        sampled = neighbors[
            np.random.choice(len(neighbors), self.n_samples)
        ]

        noise = np.random.normal(
            scale=self.noise_scale,
            size=sampled.shape
        )

        return sampled + noise

    def _kernel(self, x, Z):
        dists = euclidean_distances(Z, x.reshape(1, -1)).ravel()
        sigma = np.std(dists) + 1e-8
        weights = np.exp(-(dists ** 2) / (sigma ** 2))
        return weights

    def explain(self, x, surrogate="linear"):
        """
        Explain the local behavior of delta_fn around x
        """
        if self.X_train is None or self.delta_fn is None:
            raise ValueError("Call fit() before explain().")

        # 1. sample
        Z = self._sample_locally(x)

        # 2. evaluate delta function
        y = self.delta_fn(Z)

        if len(y.shape) > 1:
            y = y.ravel()

        # 3. weights
        weights = self._kernel(x, Z)

        # 4. surrogate model
        if surrogate == "linear":
            surrogate_model = Ridge(alpha=1.0)
        elif surrogate == "rf":
            surrogate_model = RandomForestRegressor(max_depth=5)
        else:
            raise ValueError("Unsupported surrogate type")

        surrogate_model.fit(Z, y, sample_weight=weights)

        return surrogate_model, Z, weights

    def explain_instance(self, x):
        """
        Returns the actual delta at x
        """
        return self.delta_fn(x.reshape(1, -1))[0]

    def get_feature_importance(self, surrogate_model):
        if hasattr(surrogate_model, "coef_"):
            return surrogate_model.coef_
        elif hasattr(surrogate_model, "feature_importances_"):
            return surrogate_model.feature_importances_
        else:
            return None
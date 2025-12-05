import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple


def generate_synthetic_data(
    n_samples_per_class: int = 200, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple 2D binary classification dataset using two Gaussian blobs.
    """
    rng = np.random.default_rng(random_state)

    mean0 = np.array([-1.0, -1.0])
    mean1 = np.array([1.0, 1.0])
    cov = np.array([[0.4, 0.1], [0.1, 0.4]])

    X0 = rng.multivariate_normal(mean0, cov, size=n_samples_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_samples_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate(
        [np.zeros(n_samples_per_class, dtype=int), np.ones(n_samples_per_class, dtype=int)]
    )

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    indices = rng.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    # Numerically stable softmax
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    m = y.shape[0]
    oh = np.zeros((m, n_classes))
    oh[np.arange(m), y] = 1.0
    return oh


@dataclass
class NeuralNetworkConfig:
    n_inputs: int
    n_hidden: int
    n_outputs: int
    learning_rate: float = 0.1
    random_state: int = 42


class NeuralNetwork:
    """
    Simple fully-connected neural network with:
    - One hidden layer
    - ReLU activation in the hidden layer
    - Softmax output layer
    - Cross-entropy loss
    Implemented using only NumPy.
    """

    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        rng = np.random.default_rng(config.random_state)

        # Weight initialisation (small random values)
        self.W1 = rng.normal(0, 0.1, size=(config.n_inputs, config.n_hidden))
        self.b1 = np.zeros((1, config.n_hidden))

        self.W2 = rng.normal(0, 0.1, size=(config.n_hidden, config.n_outputs))
        self.b2 = np.zeros((1, config.n_outputs))

    def _forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform a forward pass through the network.
        Returns a dictionary with intermediate values used in backprop.
        """
        z1 = X @ self.W1 + self.b1        # (m, n_hidden)
        a1 = relu(z1)                     # (m, n_hidden)
        z2 = a1 @ self.W2 + self.b2       # (m, n_outputs)
        y_hat = softmax(z2)               # (m, n_outputs)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return cache

    def _compute_loss(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        """
        Cross-entropy loss for multi-class classification.
        y_true are integer class labels.
        """
        m = y_true.shape[0]
        # Avoid log(0)
        eps = 1e-15
        correct_log_probs = -np.log(y_hat[np.arange(m), y_true] + eps)
        loss = np.sum(correct_log_probs) / m
        return loss

    def _backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation to compute gradients of the loss w.r.t. parameters.
        """
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        y_hat = cache["y_hat"]

        m = X.shape[0]
        n_classes = self.config.n_outputs

        # One-hot encode true labels
        Y = one_hot(y_true, n_classes)

        # Gradient of loss w.r.t. z2 (softmax + cross-entropy combined)
        dz2 = (y_hat - Y) / m  # (m, n_outputs)

        dW2 = a1.T @ dz2       # (n_hidden, n_outputs)
        db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, n_outputs)

        # Backprop into hidden layer
        da1 = dz2 @ self.W2.T               # (m, n_hidden)
        dz1 = da1 * relu_derivative(z1)     # (m, n_hidden)

        dW1 = X.T @ dz1                     # (n_inputs, n_hidden)
        db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, n_hidden)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def _update_parameters(self, grads: Dict[str, np.ndarray]) -> None:
        lr = self.config.learning_rate
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 1000, verbose: bool = True) -> None:
        for epoch in range(1, n_epochs + 1):
            cache = self._forward(X)
            loss = self._compute_loss(cache["y_hat"], y)
            grads = self._backward(cache, y)
            self._update_parameters(grads)

            if verbose and epoch % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.3f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cache = self._forward(X)
        return cache["y_hat"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


def plot_decision_boundary(model: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> None:
    """
    Visualise the decision boundary learned by the network for 2D data.
    """
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.title("Decision Boundary of Neural Network (2D)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Generate data
    X, y = generate_synthetic_data(n_samples_per_class=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    # 2. Create and train model
    config = NeuralNetworkConfig(
        n_inputs=2,
        n_hidden=8,
        n_outputs=2,
        learning_rate=0.1,
        random_state=42,
    )

    model = NeuralNetwork(config)
    model.fit(X_train, y_train, n_epochs=1000, verbose=True)

    # 3. Evaluate on test set
    test_acc = model.accuracy(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.3f}")

    # 4. Plot decision boundary (using all data)
    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()

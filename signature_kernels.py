import gpjax and gpx
import jax.numpy as jnp
from gpjax.kernels import AbstractKernel
from gpjax.parameters import PositiveReal

from custom_classes_and_functions import CustomComputeEngine
from signature_algorithms import signature_kernel_algorithm_compiled



class SignatureKernel_RBF(gpx.kernels.AbstractKernel):
    '''
    Custom class for the kernel, which implements the signature kernel with static RBF lift.

    ***Attributes***
    *n_levels* : The number of non-trivial (the first signature level is considered trivial) levels in the Signature kernel. 
    *lengthscales* : The lengthscale parameters of the static RBF lift
    *amplitude* : The amplitude parameter of the static RBF lift
    *weights* : The parameters responsible for the weighting of each of the signature levels - should have length (n_levels+1)
    *kernel_training* : Boolean controlling whether the above parameters are to be optimised during training
    '''
    def __init__(self, n_levels, lengthscales, amplitude, weights, kernel_training=True, compute_engine=CustomComputeEngine()):
        super().__init__(compute_engine=compute_engine)
        self.n_levels = n_levels
        self.lengthscales = PositiveReal(value=lengthscales, tag="lengthscales", trainable=kernel_training)
        self.amplitude = PositiveReal(value=amplitude, tag="amplitude", trainable=kernel_training)
        self.weights = PositiveReal(value=weights, tag="weights", trainable=kernel_training)

    def incorporate_lengthscales(self, X):
        '''
        Args:
            X:  (n_sequences, n_dimensions, len_sequences)

        Returns:
            X_scaled: (n_sequences, n_dimensions, len_sequences)
        '''
        X_scaled = X / self.lengthscales[None, :, None]
        return X_scaled

    def rbf_kernel(self, X, X2=None):
        """
        Computes RBF kernel between all time steps in X and X2.
        
        Args:
            X:  (n_sequences, n_dimensions, len_sequences)
            X2: (n_sequences2, n_dimensions2, len_sequences2)

        Returns:
            K: (n_sequences, n_sequences2, len_sequences, len_sequences2)
        """
        n_sequences, d, len_sequences = X.shape
        signal_var = self.amplitude ** 2

        if X2 is None:
            dist = square_dist(X)
            K = signal_var * jnp.exp(-dist / 2)
        else:
            n_sequences2, d2, len_sequences2 = X2.shape
            assert d2 == d, "Dimension mismatch between X and X2"
            dist = square_dist(X, X2)
            K = signal_var * jnp.exp(-dist / 2)
        return K
        
    def __call__(self, X, X2=None):
        X_scaled = self.incorporate_lengthscales(X)
        X2_scaled = self.incorporate_lengthscales(X2) if X2 is not None else None

        M = self.rbf_kernel(X_scaled, X2_scaled)

        kernel_levels = signature_kernel_algorithm_compiled(M, self.n_levels)
        SignatureKernel = jnp.tensordot(self.weights, kernel_levels, axes=(0, 0))

        return SignatureKernel



def square_dist(X, X2=None):
    """
    Computes squared Euclidean distance between all pairs of time steps in X and X2.

    Args:
        X:  (n_sequences, n_dimensions, len_sequences)
        X2: (n_sequences2, n_dimensions2, len_sequences2)

    Returns:
        dist: (n_sequences, n_sequences2, len_sequences, len_sequences2)
    """
    n_sequences, d, len_sequences = X.shape
    X = jnp.transpose(X, (0, 2, 1))  # (n_sequences, len_sequences, d)
    X_flat = X.reshape(-1, d)        # (n_sequences * len_sequences, d)

    if X2 is None:
        X2_flat = X_flat
    else:
        n_sequences2, d2, len_sequences2 = X2.shape
        assert d2 == d, "Dimension mismatch between X and X2"
        X2 = jnp.transpose(X2, (0, 2, 1))  # (n_sequences2, len_sequences2, d2)
        X2_flat = X2.reshape(-1, d)        # (n_sequences2 * len_sequences2, d2) 

    Xs = jnp.sum(X_flat**2, axis=-1)
    X2s = jnp.sum(X2_flat**2, axis=-1)

    dist = -2 * jnp.matmul(X_flat, X2_flat.T)
    dist += Xs[:, None] + X2s[None, :]

    if X2 is None:
        dist = dist.reshape(n_sequences, len_sequences, n_sequences, len_sequences)
    else:
        dist = dist.reshape(n_sequences, len_sequences, n_sequences2, len_sequences2)

    dist = jnp.transpose(dist, (0, 2, 1, 3))  # (n_sequences, n_sequences2, len_sequences, len_sequences2)
    return dist

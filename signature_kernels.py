
from .custom_classes_and_functions import CustomComputeEngine

class SignatureKernel(gpx.kernels.AbstractKernel):
    def __init__(self, n_levels, lengthscales, weights, active_dims=None, n_dims=None, compute_engine=CustomComputeEngine()):
        super().__init__(active_dims=None, n_dims=None, compute_engine=compute_engine)
        self.n_levels = n_levels
        self.lengthscales = Parameter(value=lengthscales, tag="lengthscales")
        self.weights = Parameter(value=weights, tag="weights")
        self.static_kernel = self.trivial_kernel

    def incorporate_lengthscales(self, X):
        X_scaled = X / self.lengthscales[None, None, :]
        return X_scaled

    def __call__(self, X, X2=None):

        X_scaled = self.incorporate_lengthscales(X)
        X2_scaled = self.incorporate_lengthscales(X2) if X2 is not None else None

        M = self.static_kernel(X_scaled, X2_scaled)
        kernel_levels = signature_kernel_algorithm_compiled(M, self.n_levels)
        K = jnp.tensordot(self.weights, kernel_levels, axes=(0, 0))

        return K

    def trivial_kernel(self, X, X2):
        ''' Computes the most trivial choice of kernel between paths
        Args:
          X and Y must have the shape (batch_size, length, dimensions)
        '''
        return jnp.matmul(X, jnp.swapaxes(X2, -1, -2))


class SignatureKernel_RBF(SignatureKernel):
    def __init__(self, n_levels, lengthscales, weights, amplitude, active_dims=None, n_dims=None, compute_engine=CustomComputeEngine()):
        SignatureKernel.__init__(self, n_levels, lengthscales, weights, active_dims=None, n_dims=None, compute_engine=compute_engine)
        self.static_kernel = self.rbf_kernel
        self.amplitude = Parameter(value=amplitude, tag="amplitude")

    def rbf_kernel(self, X, X2=None):
        """
        Computes RBF kernel between all time steps in X and X2 (flattened view),
        and reshapes into: (num_examples, num_examples2, len_examples, len_examples2)
        """
        num_examples, len_examples, d = X.shape
        signal_var = self.amplitude ** 2
        if X2 is None:
            dist = square_dist(X)
            K = signal_var * jnp.exp(-dist / 2)
            K = K.reshape(num_examples, len_examples, num_examples, len_examples)
            K = jnp.transpose(K, (0, 2, 1, 3))  # (num_ex, num_ex, len_ex, len_ex)
        else:
            num_examples2, len_examples2, _ = X2.shape
            dist = square_dist(X, X2)
            K = signal_var * jnp.exp(-dist / 2)
            K = K.reshape(num_examples, len_examples, num_examples2, len_examples2)
            K = jnp.transpose(K, (0, 2, 1, 3))  # (num_ex, num_ex2, len_ex, len_ex2)

        return K


def square_dist(X, X2=None):
    """
    X:  (num_examples, len_examples, d)
    X2: (num_examples2, len_examples2, d), optional
    
    Returns:
    D2: (num_examples * len_examples, num_examples2 * len_examples2)
    """
    num_examples, len_examples, d = X.shape
    X_flat = X.reshape(-1, d)  # (num_examples * len_examples, d)
    
    if X2 is None:
        X2_flat = X_flat
    else:
        X2_flat = X2.reshape(-1, d)
    
    Xs = jnp.sum(X_flat**2, axis=-1)  # (N,)
    X2s = jnp.sum(X2_flat**2, axis=-1)  # (M,)
    
    dist = -2 * jnp.matmul(X_flat, X2_flat.T)  # (N, M)
    dist += Xs[:, None] + X2s[None, :]         # (N, 1) + (1, M)
    
    return dist

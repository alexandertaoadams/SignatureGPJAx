import typing as tp
import cola
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from flax import nnx
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import Transform
from scipy.optimize import minimize

from gpjax.dataset import Dataset
from gpjax.objectives import Objective
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)
from gpjax.scan import vscan
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

Model = tp.TypeVar("Model", bound=nnx.Module)



class CustomDataset(gpx.dataset.Dataset):
    '''Custom class for the dataset, in the case where inputs are 3-dimensional rather than 2-dimensional:
    (n_sequences, n_dimensions, seq_length)
    '''
    def __init__(self,X,y):
        self.X = X
        self.y = y



class CustomComputeEngine(gpx.kernels.computations.AbstractKernelComputation):
    '''Custom class for the computation engine, in the case where inputs are 3-dimensional rather than 2-dimensional:
    (n_sequences, n_dimensions, seq_length)
    '''
    def gram(self, kernel, X):
        return cola.PSD(cola.ops.Dense(kernel(X, X)))

    def cross_covariance(self, kernel, X, X2):
        return cola.PSD(cola.ops.Dense(kernel(X,X2))

    def diagonal(self, kernel, X):
        return cola.PSD(cola.ops.Dense(kernel(X))




def custom_fit(  # noqa: PLR0913
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    key: KeyArray = jr.PRNGKey(42),
    num_iters: int = 100,
    batch_size: int = -1,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
) -> tuple[Model, jax.Array]:
    r"""Train a Module model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Example:
    ```pycon
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>> from gpjax.parameters import PositiveReal, Static
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>> # (2) Define your model:
        >>> class LinearModel(nnx.Module):
        >>>     def __init__(self, weight: float, bias: float):
        >>>         self.weight = PositiveReal(weight)
        >>>         self.bias = Static(bias)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.weight.value * x + self.bias.value
        >>>
        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> def mse(model, data):
        >>>     pred = model(data.X)
        >>>     return jnp.mean((pred - data.y) ** 2)
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
        >>>     model=model, objective=mse, train_data=D, optim=ox.sgd(0.001), num_iters=1000
        >>> )
    ```

    Args:
        model (Model): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (int): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (int): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (KeyArray): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (int): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (bool): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns:
        A tuple comprising the optimised model and training history.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, Parameter, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params: nnx.State, batch: Dataset) -> ScalarFloat:
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, batch)

    # Initialise optimiser state.
    opt_state = optim.init(params)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = _custom_get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (params, _), history = scan(step, (params, opt_state), (iter_keys), unroll=unroll)

    # Parameters bijection to constrained space
    if params_bijection is not None:
        params = transform(params, params_bijection)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history



def _custom_get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    return CustomDataset(X=x[indices], y=y[indices])


def _check_model(model: tp.Any) -> None:
    """Check that the model is a subclass of nnx.Module."""
    if not isinstance(model, nnx.Module):
        raise TypeError(
            "Expected model to be a subclass of nnx.Module. "
            f"Got {model} of type {type(model)}."
        )


def _check_train_data(train_data: tp.Any) -> None:
    """Check that the train_data is of type gpjax.Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError(
            "Expected train_data to be of type gpjax.Dataset. "
            f"Got {train_data} of type {type(train_data)}."
        )


def _check_optim(optim: tp.Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError(
            "Expected optim to be of type optax.GradientTransformation. "
            f"Got {optim} of type {type(optim)}."
        )


def _check_num_iters(num_iters: tp.Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError(
            "Expected num_iters to be of type int. "
            f"Got {num_iters} of type {type(num_iters)}."
        )

    if num_iters <= 0:
        raise ValueError(f"Expected num_iters to be positive. Got {num_iters}.")


def _check_log_rate(log_rate: tp.Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError(
            "Expected log_rate to be of type int. "
            f"Got {log_rate} of type {type(log_rate)}."
        )

    if not log_rate > 0:
        raise ValueError(f"Expected log_rate to be positive. Got {log_rate}.")


def _check_verbose(verbose: tp.Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError(
            "Expected verbose to be of type bool. "
            f"Got {verbose} of type {type(verbose)}."
        )


def _check_batch_size(batch_size: tp.Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError(
            "Expected batch_size to be of type int. "
            f"Got {batch_size} of type {type(batch_size)}."
        )

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError(f"Expected batch_size to be positive or -1. Got {batch_size}.")


__all__ = [
    "fit",
    "_custom_get_batch",
]

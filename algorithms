import jax
import jax.numpy as jnp
from jax import lax


def signature_kernel_algorithm(M, n_levels: int, order: int = 3,
                                difference: bool = True,
                                return_levels: bool = True):


    def compute_R_next(M, R, d, order):

        def fill_entry(R_next, r, s, val):
            return R_next.at[r, s].set(val)

        R_next = jnp.zeros((order, order) + M.shape, dtype=M.dtype)

        R_next = fill_entry(
            R_next, 0, 0,
            M * multi_cumsum(jnp.sum(R, axis=(0, 1)), exclusive=True, axis=(-2, -1)))


        def row_body(r, R_next):
            R_next = fill_entry(
                R_next, 0, r,
                1. / (r + 1) * M * multi_cumsum(jnp.sum(R[:, r - 1], axis=0), exclusive=True, axis=-2)
            )
            R_next = fill_entry(
                R_next, r, 0,
                1. / (r + 1) * M * multi_cumsum(jnp.sum(R[r - 1, :], axis=0), exclusive=True, axis=-1)
            )

            def col_body(s, R_next):
                val = 1. / ((r + 1) * (s + 1)) * M * R[r - 1, s - 1]
                return fill_entry(R_next, r, s, val)

            R_next = jax.lax.fori_loop(1, d, col_body, R_next)
        return R_next


    if difference:
      M = jnp.diff(jnp.diff(M, axis=-2), axis=-1)

    if M.ndim == 4:
      n_X, n_Y = M.shape[0], M.shape[1]
      K_init = jnp.ones((n_X, n_Y), dtype=M.dtype)
    else:
      n_X = M.shape[0]
      K_init = jnp.ones((n_X,), dtype=M.dtype)

    if return_levels:
      levels = [K_init, jnp.sum(M, axis=(-2, -1))]
    else:
      levels = None
      K_init += jnp.sum(M, axis=(-2, -1))

    def init_R0(M, order):
      R0 = jnp.zeros((order, order) + M.shape, dtype=M.dtype)
      R0 = R0.at[0, 0].set(M)
      return R0

    R0 = init_R0(M, order)

    def dynamic_iteration_needs_scan(carry, i):
      R_prev, K_prev = carry
      d = jnp.minimum(i+1, order)
      R_next = compute_R_next(M, R_prev, d, order)
      R_sum = jnp.sum(R_next, axis=(0, 1, -2, -1))
      K_new = K_prev + R_sum
      return (R_next, K_new), R_sum

    (final_R, final_K), level_values = lax.scan(dynamic_iteration_needs_scan,
                                                (R0, K_init),
                                                jnp.arange(1, n_levels))

    if return_levels:
      return jnp.stack([levels[0], levels[1], *level_values], axis=0)
    else:
      return final_K


signature_kernel_algorithm_compiled = jax.jit(
    signature_kernel_algorithm,
    static_argnames=('n_levels', 'order', 'difference', 'return_levels')
)

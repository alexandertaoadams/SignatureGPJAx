

def signature_kernel_algorithm(S, n_levels):
    ''' 
    Function that computes the signature kernel from the lifted datapoints.

    Args:
        S: Matrix of static kernel lifted data, with dimensions (n_sequences, n_sequences2, seq_length, seq_length2)
        n_levels: Number of nontrivial levels (the zeroth level is considered trivial) in the signature kernel
   
    Returns:
        Array of length (n_levels + 1) consisting of the kernel levels (including the zeroth level).
        Each kernel level has dimensions (n_sequences, n_sequences2).
    '''

    def modified_cumsum(X, axis=-1):
        '''Like jnp.cumsum, but can be over multiple axes'''
        ndim = X.ndim
        axis = [axis] if jnp.isscalar(axis) else axis
        axis = [ndim + ax if ax < 0 else ax for ax in axis]

        slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
        X = X[slices]

        for ax in axis:
            X = jnp.cumsum(X, axis=ax)

        pads = tuple((1, 0) if ax in axis else (0, 0) for ax in range(ndim))
        X = jnp.pad(X, pads)

        return X

    def compute_A_next(S, A_prev, idx, max_shape):
        def fill_entry(A, a, b, val):
            return A.at[a, b].set(val)

        A_next = jnp.zeros(max_shape, dtype=S.dtype)
        A_next = fill_entry(A_next, 0, 0, S * modified_cumsum(jnp.sum(A_prev, axis=(0, 1)), axis=(-2, -1)))

        def row_step(a, A_next):
            A_next = fill_entry(
                A_next, 0, a,
                1.0 / (a + 1) * S * modified_cumsum(jnp.sum(A_prev[:, a - 1], axis=0), axis=-2)
            )
            A_next = fill_entry(
                A_next, a, 0,
                1.0 / (a + 1) * S * modified_cumsum(jnp.sum(A_prev[a - 1, :], axis=0), axis=-1)
            )

            def col_step(b, A_next):
                val = 1.0 / ((a + 1) * (b + 1)) * S * A_prev[a - 1, b - 1]
                return fill_entry(A_next, a, b, val)

            A_next = jax.lax.fori_loop(1, idx, col_step, A_next)
            return A_next

        A_next = jax.lax.fori_loop(1, idx, row_step, A_next)
        return A_next


    def recurrence_step_for_scanning(A_prev, idx):
        A_next = compute_A_next(S, A_prev, idx, max_shape)
        L_next = jnp.sum(A_next, axis=(0, 1, -2, -1))
        return A_next, L_next


    # Apply second-order differencing to static lift values
    S = jnp.diff(jnp.diff(S, axis=-2), axis=-1)

    # We can calculate the zeroth and first levels of the signature kernel easily
    levels = [jnp.ones((S.shape[0], S.shape[1]), dtype=S.dtype), jnp.sum(S, axis=(-2, -1))]

    # Initialise A_0
    max_shape = (n_levels, n_levels) + S.shape
    A_0 = jnp.zeros(max_shape, dtype=S.dtype).at[0, 0].set(S)

    # Compute remaining levels
    _, level_values = lax.scan(
    recurrence_step_for_scanning,
    A_0,
    jnp.arange(2, n_levels + 1)
    )

    # Return signature levels as array
    return jnp.stack([levels[0], levels[1], *level_values], axis=0)


signature_kernel_algorithm_compiled = jax.jit(signature_kernel_algorithm, static_argnames=["n_levels"])

import jax.numpy as jnp
from jax import random


def dropout(x: jnp.ndarray, key, p=0.5, train=True):
    '''
    how to use:

    >>>>>> x, key = dropout(x, key, p=0.5, self.train)
    '''
    if not train:
        return jnp.arange(10), x, key

    p_keep = 1 - p
    mask = random.bernoulli(key, p_keep, x.shape)
    new_key, _ = random.split(key)  # update key, to make mask different in different **batch**.

    return mask, jnp.where(mask, x / p_keep, 0), new_key  # scale here to make E(X) the same while evaluating.

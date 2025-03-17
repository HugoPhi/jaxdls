'''
JAX Neural Network Utilities Module

* Last Updated: 2025-03-09
* Author: HugoPhi, [GitHub](https://github.com/HugoPhi)
* Maintainer: hugonelsonm3@gmail.com


This module provides essential utilities for building neural networks using JAX,
including dropout regularization and linear transformations. These utilities are
designed to be lightweight, efficient, and compatible with JAX's functional
programming paradigm.

Key Features:
    - Dropout regularization for preventing overfitting during training
    - Linear layer implementation for dense transformations
    - Support for JAX's random key management
    - Training/evaluation mode handling for dropout

Structure:
    - Dropout function with training/evaluation mode support
    - Linear layer implementation with weight and bias application
    - Configuration generator for linear layers

Typical Usage:
    1. Use `dropout` during training for regularization
    2. Create linear layers with `get_linear` and apply them with `linear`
    3. Manage random keys properly for reproducible dropout masks

Note: All functions are designed to work seamlessly with JAX's automatic
differentiation and hardware acceleration capabilities.
'''

import jax.numpy as jnp
from jax import random


def dropout(key, x: jnp.ndarray, p=0.5, train=True):
    '''
    Applies dropout to the input array during training.

    Dropout is a regularization technique that randomly sets a fraction of the input
    elements to zero during training. This helps prevent overfitting by reducing the
    co-dependency between neurons.

    During evaluation (i.e., when `train=False`), dropout is disabled, and the input
    is returned unchanged. During training, the output is scaled by `1 / (1 - p)` to
    maintain the expected value of the input.

    Args:
        x: Input array of any shape.
        key: A JAX random key for generating random values.
        p: Dropout probability (default: 0.5). Each element of the input is set to zero
           with probability `p`.
        train: Whether the model is in training mode (default: True). If `False`, dropout
               is disabled, and the input is returned unchanged.

    Returns:
        x_out: The output array after applying dropout. During training, some elements
               are set to zero, and the remaining elements are scaled by `1 / (1 - p)`.
               During evaluation, the input is returned unchanged.
        new_key: A new JAX random key, updated to ensure different masks are generated
                 for different batches.

    Examples
    --------
    >>> x = jnp.ones((10, 10))  # Example input
    >>> key = random.PRNGKey(0)  # Random key
    >>> x_out, key = dropout(x, key, p=0.5, train=True)  # Apply dropout during training
    '''

    if not train:
        return x, key

    p_keep = 1 - p
    new_key, use_key = random.split(key)  # update key, to make mask different in different **batch**.
    mask = random.bernoulli(use_key, p_keep, x.shape)

    return new_key, jnp.where(mask, x / p_keep, 0)  # scale here to make E(X) the same while evaluating.


def _linear(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray):
    '''
    Computes the output of a linear layer.

    Args:
        x: Input tensor of shape (batch_size, in_features).
        w: Weight matrix of shape (out_features, in_features).
        b: Bias vector of shape (out_features,).

    Returns:
        Output tensor of shape (batch_size, out_features).
    '''

    return jnp.dot(x, w) + b


def get_linear(input_dim, output_dim, strategy='Kaiming'):
    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'strategy': strategy,
    }


def linear(x, params):
    '''
    Computes the output of a linear layer.

    Input:
        x: Input tensor of shape (batch_size, in_features).

    Output:
        res: (batch_size, out_features)
    '''

    w = params['w']
    b = params['b']
    return _linear(x, w, b)

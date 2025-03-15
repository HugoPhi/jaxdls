'''
JAX Convolution and Pooling Operations Module

* Last Updated: 2025-03-09
* Author: HugoPhi, [GitHub](https://github.com/HugoPhi)
* Maintainer: hugonelsonm3@gmail.com

This module provides optimized implementations of multi-dimensional convolution
and max-pooling operations using JAX's accelerated numerical computing framework.
Contains 1D/2D/3D variants for both convolution and pooling operations, designed
specifically for neural network implementations requiring high-performance computation.

Key Features:
    - 2D/3D/1D convolution operations with configurable padding
    - Corresponding max-pooling operations with adjustable window/strides
    - Parameter management utilities for layer configuration
    - JAX backend utilization for GPU/TPU acceleration

Structure:
    - Private implementations (_convNd, _max_poolingNd) handling core computation
    - Public interfaces (convNd, max_poolingNd) for end-user interaction
    - Configuration generators (get_convNd, get_max_poolNd) for hyperparameter management

Typical Usage:
    1. Generate layer config with get_convNd()/get_max_poolNd()
    2. Initialize parameters (weights/biases) matching config specs
    3. Execute forward pass through convNd()/max_poolingNd()

Note: All tensor shapes follow channel-first format (NCHW for 2D, NCW for 1D,
NCDHW for 3D). Requires JAX installation and compatible hardware for acceleration.
'''


import jax.numpy as jnp
from jax import lax


def _conv2d(x, w, b, padding=1):
    '''
    Performs a 2D convolution operation using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, I, H, W).
        w: Convolution kernel of shape (O, I, KH, KW).
        b: Bias term of shape (O,).
        padding: Padding size (default: 1).

    Returns:
        res: Output tensor of shape (B, O, H', W').
    '''

    dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
    padding_mode = ((padding, padding), (padding, padding))  # 高度和宽度方向的padding

    out = lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=(1, 1),
        padding=padding_mode,
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
        dimension_numbers=dimension_numbers
    )

    return out + b[None, :, None, None]


def get_conv2d(input_channels, output_channels, kernel_size, padding=0, strategy='Kaiming'):
    '''
    Returns a dictionary of hyperparameters for a 2D convolution layer.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_size: Kernel size. such as: (2, 2).
        padding: Padding size (default: 0).
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'input_channel': input_channels,
        'output_channel': output_channels,
        'kernel_size': kernel_size,
        'padding': padding,
        'strategy': strategy,
    }


def conv2d(x, params, config):
    '''
    Performs a 2D convolution operation using JAX's optimized functions.

    Input:
        x: Input tensor of shape (B, I, H, W).

    Returns:
        res: (B, O, H', W').
    '''

    w = params['w']
    b = params['b']
    padding = config['padding']
    return _conv2d(x, w, b, padding=padding)


def _max_pooling2d(x, pool_size=(2, 2), stride=None):
    '''
    Performs 2D max pooling using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, C, H, W).
        pool_size: Pooling window size (default: (2, 2)).
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        res: Output tensor of shape (B, C, H', W').
    '''

    if stride is None:
        stride = pool_size

    return lax.reduce_window(
        operand=x,
        init_value=-jnp.inf,
        computation=lax.max,
        window_dimensions=(1, 1, pool_size[0], pool_size[1]),
        window_strides=(1, 1, stride[0], stride[1]),
        padding='VALID'
    )


def get_max_pool2d(pool_size, stride=None):
    '''
    Returns a dictionary of hyperparameters for a 2D max pooling layer.

    Args:
        pool_size: Pooling window size. (e.g. pool_size=(2, 2))
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'pool_size': pool_size,
        'stride': stride,
    }


def max_pooling2d(x, config):
    '''
    Performs 2D max pooling using JAX's optimized functions.

    Input:
        x: (B, C, H, W).

    Output:
        res: (B, C, H', W').
    '''

    pool_size = config['pool_size']
    stride = config['stride']
    return _max_pooling2d(x, pool_size, stride)


def _conv1d(x, w, b, padding=1):
    '''
    Performs a 1D convolution operation using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, I, L).
        w: Convolution kernel of shape (O, I, K).
        b: Bias term of shape (O,).
        padding: Padding size (default: 1).

    Returns:
        res: Output tensor of shape (B, O, L').
    '''

    dimension_numbers = ('NCW', 'OIW', 'NCW')
    return lax.conv_general_dilated(
        x, w, (1,), [(padding, padding)],
        (1,), (1,), dimension_numbers
    ) + b[None, :, None]


def get_conv1d(input_channels, output_channels, kernel_size, padding=0, strategy='Kaiming'):
    '''
    Returns a dictionary of hyperparameters for a 1D convolution layer.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_size: Kernel size.
        padding: Padding size (default: 0).
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'input_channel': input_channels,
        'output_channel': output_channels,
        'kernel_size': kernel_size,
        'padding': padding,
        'strategy': strategy,
    }


def conv1d(x, params, config):
    '''
    Performs a 1D convolution operation using JAX's optimized functions.

    Input:
        x: Input tensor of shape (B, I, L).

    Returns:
        res: (B, O, L').
    '''
    w = params['w']
    b = params['b']
    padding = config['padding']
    return _conv1d(x, w, b, padding=padding)


def _max_pooling1d(x, pool_size=2, stride=None):
    '''
    Performs 1D max pooling using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, C, L).
        pool_size: Pooling window size (default: 2).
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        res: Output tensor of shape (B, C, L').
    '''

    if stride is None:
        stride = pool_size

    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        (1, 1, pool_size), (1, 1, stride),
        'VALID'
    )


def get_max_pool1d(pool_size, stride=None):
    '''
    Returns a dictionary of hyperparameters for a 1D max pooling layer.

    Args:
        pool_size: Pooling window size.
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'pool_size': pool_size,
        'stride': stride,
    }


def max_pooling1d(x, config):
    '''
    Performs 1D max pooling using JAX's optimized functions.

    Input:
        x: (B, C, L).

    Output:
        res: (B, C, L').
    '''

    pool_size = config['pool_size']
    stride = config['stride']
    return _max_pooling1d(x, pool_size, stride)


def _conv3d(x, w, b, padding=1):
    '''
    Performs a 3D convolution operation using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, I, D, H, W).
        w: Convolution kernel of shape (O, I, KD, KH, KW).
        b: Bias term of shape (O,).
        padding: Padding size (default: 1).

    Returns:
        res: Output tensor of shape (B, O, D', H', W').
    '''

    dimension_numbers = ('NCDHW', 'OIDHW', 'NCDHW')
    padding = [(padding, padding)] * 3
    return lax.conv_general_dilated(
        x, w, (1, 1, 1), padding,
        (1, 1, 1), (1, 1, 1), dimension_numbers
    ) + b[None, :, None, None, None]


def get_conv3d(input_channels, output_channels, kernel_size, padding=0, strategy='Kaiming'):
    '''
    Returns a dictionary of hyperparameters for a 3D convolution layer.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_size: Kernel size.
        padding: Padding size (default: 0).
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'input_channel': input_channels,
        'output_channel': output_channels,
        'kernel_size': kernel_size,
        'padding': padding,
        'strategy': strategy,
    }


def conv3d(x, params, config):
    '''
    Performs a 3D convolution operation using JAX's optimized functions.

    Input:
        x: Input tensor of shape (B, I, D, H, W).

    Returns:
        res: (B, O, D', H', W').
    '''
    w = params['w']
    b = params['b']
    padding = config['padding']
    return _conv3d(x, w, b, padding=padding)


def _max_pooling3d(x, pool_size=(2, 2, 2), stride=None):
    '''
    Performs 3D max pooling using JAX's optimized functions.

    Args:
        x: Input tensor of shape (B, C, D, H, W).
        pool_size: Pooling window size (default: (2, 2, 2)).
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        res: Output tensor of shape (B, C, D', H', W').
    '''

    if stride is None:
        stride = pool_size

    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        (1, 1, pool_size[0], pool_size[1], pool_size[2]),
        (1, 1, stride[0], stride[1], stride[2]),
        'VALID'
    )


def get_max_pool3d(pool_size, stride=None):
    '''
    Returns a dictionary of hyperparameters for a 3D max pooling layer.

    Args:
        pool_size: Pooling window size.
        stride: Stride for the pooling operation (default: None, same as pool_size).

    Returns:
        hyperparameters: Dictionary of hyperparameters.
    '''

    return {
        'pool_size': pool_size,
        'stride': stride,
    }


def max_pooling3d(x, config):
    '''
    Performs 3D max pooling using JAX's optimized functions.

    Input:
        x: (B, C, D, H, W).

    Output:
        res: (B, C, D', H', W').
    '''

    pool_size = config['pool_size']
    stride = config['stride']
    return _max_pooling3d(x, pool_size, stride)

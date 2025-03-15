'''
JAX Recurrent Neural Network (RNN) Module

* Last Updated: 2025-03-09
* Author: HugoPhi, [GitHub](https://github.com/HugoPhi)
* Maintainer: hugonelsonm3@gmail.com


This module provides optimized implementations of recurrent neural network (RNN)
variants, including Basic RNN, LSTM, and GRU cells, using JAX's `lax.scan` for
efficient sequence processing. Designed for high-performance sequence modeling
tasks, these implementations leverage JAX's acceleration capabilities for
GPU/TPU compatibility.

Key Features:
    - Basic RNN, LSTM, and GRU cell implementations
    - Memory-efficient sequence processing via `lax.scan`
    - Configurable hyperparameters for flexible architecture design
    - Support for batch processing and variable sequence lengths

Structure:
    - Private cell implementations (_basic_rnn_cell, _lstm_cell, _gru_cell) for core logic
    - Public interfaces (basic_rnn, lstm, gru) for end-user interaction
    - Configuration generators (get_basic_rnn, get_lstm, get_gru) for hyperparameter management

Typical Usage:
    1. Generate layer config with appropriate get_* function
    2. Initialize parameters matching config specs
    3. Execute forward pass through corresponding RNN variant

Note: All implementations assume input shape (S, B, I) where:
    - S: Sequence length
    - B: Batch size
    - I: Input dimension
Hidden states are of shape (B, H) where H is the hidden dimension.
'''


import jax.numpy as jnp
from jax import lax
from ...utils import sigmoid


def _basic_rnn_cell(x, h0,
                    w_hh, w_xh, b_h,
                    w_hy, b_y):
    '''
    Implements a basic RNN cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        h0: Initial hidden state of shape (B, H), where:
            - H: Hidden state dimension
        w_hh: Hidden-to-hidden weight matrix of shape (H, H).
        w_xh: Input-to-hidden weight matrix of shape (I, H).
        b_h: Hidden state bias of shape (H,).
        w_hy: Hidden-to-output weight matrix of shape (H, O), where:
              - O: Output dimension
        b_y: Output bias of shape (O,).

    Returns:
        res: Output sequence of shape (S, B, O).
        h: Final hidden state of shape (B, H).
    '''

    def step(carry, x_t):
        h_prev = carry

        h_new = jnp.tanh(h_prev @ w_hh + x_t @ w_xh + b_h)
        res = h_new @ w_hy + b_y

        return h_new, res

    h, res = lax.scan(step, h0, x)

    return res, h


def get_basic_rnn(timesteps, input_dim, output_dim, hidden_dim, strategy='Xavier'):
    '''
    Returns a dictionary containing the hyperparameters of the basic RNN cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_dim: Hidden state dimension.
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        A dictionary containing the hyperparameters of the basic RNN cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'strategy': strategy
    }


def basic_rnn(x, params):
    '''
    Implements a basic RNN cell using `lax.scan` for optimization.

    Input:
        x: (S, B, I)

    Output:
        res: Output sequence of shape (S, B, O).
        h: Final hidden state of shape (B, H).
    '''

    h0 = params['h0']
    w_hh = params['w_hh']
    w_xh = params['w_xh']
    b_h = params['b_h']
    w_hy = params['w_hy']
    b_y = params['b_y']
    return _basic_rnn_cell(x, h0, w_hh, w_xh, b_h, w_hy, b_y)


def _lstm_cell(x, h0, c0,
               Ws, Us, Bs):
    '''
    Implements an LSTM cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        h0: Initial hidden state of shape (B, H), where:
            - H: Hidden state dimension
        c0: Initial cell state of shape (B, H).
        Ws: Tuple of 4 weight matrices for input-to-hidden transformations, each of shape (I, H).
        Us: Tuple of 4 weight matrices for hidden-to-hidden transformations, each of shape (H, H).
        Bs: Tuple of 4 bias vectors, each of shape (H,).

    Returns:
        res: Output sequence of shape (S, B, H).
        h: Final hidden state of shape (B, H).
        c: Final cell state of shape (B, H).
    '''

    w_i, w_f, w_c, w_o = Ws  # (I, H)
    u_i, u_f, u_c, u_o = Us  # (H, H)
    b_i, b_f, b_c, b_o = Bs  # (H)

    def step(carry, x_t):
        h_prev, c_prev = carry
        II = sigmoid(x_t @ w_i + h_prev @ u_i + b_i)
        FF = sigmoid(x_t @ w_f + h_prev @ u_f + b_f)
        CC = jnp.tanh(x_t @ w_c + h_prev @ u_c + b_c)
        OO = sigmoid(x_t @ w_o + h_prev @ u_o + b_o)

        c_new = FF * c_prev + II * CC
        h_new = OO * jnp.tanh(c_new)
        res_new = OO

        return (h_new, c_new), res_new

    (h, c), res = lax.scan(step, (h0, c0), x)  # use scan to decrease RAM usage, I do not know why old version ram will increse by epochs
    return res, h, c


def get_lstm(timesteps, input_dim, hidden_dim, strategy='Xavier'):
    '''
    Returns a dictionary containing the hyperparameters of the LSTM cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        A dictionary containing the hyperparameters of the LSTM cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'strategy': strategy,
    }


def lstm(x, params, config):
    '''
    Implements an LSTM cell using `lax.scan` for optimization.

    Input:
        x: (S, B, I)

    Output:
        res: Output sequence of shape (S, B, H).
        h: Final hidden state of shape (B, H).
        c: Final cell state of shape (B, H).
    '''

    h0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    c0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    Ws = params['Ws']
    Us = params['Us']
    Bs = params['Bs']
    return _lstm_cell(x, h0, c0, Ws, Us, Bs)


def _gru_cell(x, h0,
              Ws, Us, Bs):
    '''
    Implements a GRU cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        h0: Initial hidden state of shape (B, H), where:
            - H: Hidden state dimension
        Ws: Tuple of 3 weight matrices for input-to-hidden transformations, each of shape (I, H).
        Us: Tuple of 3 weight matrices for hidden-to-hidden transformations, each of shape (H, H).
        Bs: Tuple of 3 bias vectors, each of shape (H,).

    Returns:
        res: Output sequence of shape (S, B, H).
        h: Final hidden state of shape (B, H).
    '''

    w_z, w_r, w_h = Ws  # (I, H)
    u_z, u_r, u_h = Us  # (H, H)
    b_z, b_r, b_h = Bs  # (H)

    def step(carry, x_t):
        h_prev = carry

        R = sigmoid(x_t @ w_r + h_prev @ u_r + b_r)
        Z = sigmoid(x_t @ w_z + h_prev @ u_z + b_z)

        H = jnp.tanh(x_t @ w_h + (R * h_prev) @ u_h + b_h)

        new_h = (1 - Z) * h_prev + Z * H
        return new_h, new_h

    (h), (res) = lax.scan(step, h0, x)

    return res, h


def get_gru(timesteps, input_dim, hidden_dim, strategy='Xavier'):
    '''
    Returns a dictionary containing the hyperparameters of the GRU cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier

    Returns:
        A dictionary containing the hyperparameters of the GRU cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'strategy': strategy,
    }


def gru(x, params, config):
    '''
    Implements a GRU cell using `lax.scan` for optimization, get trainable params & hyper configs

    Input:
        x: (S, B, I)

    Output:
        res: (S, B, H)
        h: Final hidden state of shape (B, H)
    '''

    h0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    Ws = params['Ws']
    Us = params['Us']
    Bs = params['Bs']
    return _gru_cell(x, h0, Ws, Us, Bs)

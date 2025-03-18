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


def _bidirectional_lstm_cell(x, h0, c0, Ws, Us, Bs):
    '''
    Implements a bidirectional LSTM cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        h0: Initial hidden state of shape (B, 2H), where:
            - H: Hidden state dimension
        c0: Initial cell state of shape (B, 2H).
        Ws: Weight matrices for input-to-hidden transformations of shape (8, I, H).
            - First 4 matrices are for forward direction
            - Last 4 matrices are for backward direction
        Us: Weight matrices for hidden-to-hidden transformations of shape (8, H, H).
            - First 4 matrices are for forward direction
            - Last 4 matrices are for backward direction
        Bs: Bias vectors of shape (8, H).
            - First 4 vectors are for forward direction
            - Last 4 vectors are for backward direction

    Returns:
        res: Output sequence of shape (S, B, 2H).
        h: Final hidden state of shape (B, 2H).
        c: Final cell state of shape (B, 2H).
    '''

    # Split parameters for forward and backward directions
    Ws_forward, Ws_backward = Ws[:4], Ws[4:]  # (4, I, H) each
    Us_forward, Us_backward = Us[:4], Us[4:]  # (4, H, H) each
    Bs_forward, Bs_backward = Bs[:4], Bs[4:]  # (4, H) each

    # Forward LSTM
    res_forward, h_forward, c_forward = _lstm_cell(
        x,
        h0[:, :h0.shape[-1] // 2],  # Use first half of h0
        c0[:, :c0.shape[-1] // 2],  # Use first half of c0
        tuple(Ws_forward),
        tuple(Us_forward),
        tuple(Bs_forward)
    )

    # Backward LSTM
    res_backward, h_backward, c_backward = _lstm_cell(
        jnp.flip(x, axis=0),
        h0[:, h0.shape[-1] // 2:],  # Use second half of h0
        c0[:, c0.shape[-1] // 2:],  # Use second half of c0
        tuple(Ws_backward),
        tuple(Us_backward),
        tuple(Bs_backward)
    )

    # Concatenate results
    return (
        jnp.concatenate([res_forward, jnp.flip(res_backward, axis=0)], axis=-1),
        jnp.concatenate([h_forward, h_backward], axis=-1),
        jnp.concatenate([c_forward, c_backward], axis=-1)
    )


def get_bilstm(timesteps, input_dim, hidden_dim, strategy='Xavier'):
    '''
    Returns a dictionary containing the hyperparameters of the bidirectional LSTM cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier.

    Returns:
        A dictionary containing the hyperparameters of the bidirectional LSTM cell.
    '''
    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'strategy': strategy,
    }


def bilstm(x, params, config):
    '''
    Implements a bidirectional LSTM cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        params: Dictionary containing the parameters:
            - Ws: Weight matrices for input-to-hidden transformations of shape (8, I, H).
            - Us: Weight matrices for hidden-to-hidden transformations of shape (8, H, H).
            - Bs: Bias vectors of shape (8, H).
        config: Dictionary containing the hyperparameters.

    Returns:
        res: Output sequence of shape (S, B, 2H).
        h: Final hidden state of shape (B, 2H).
        c: Final cell state of shape (B, 2H).
    '''
    h0 = jnp.zeros((x.shape[1], 2 * config['hidden_dim']))  # (B, 2H)
    c0 = jnp.zeros((x.shape[1], 2 * config['hidden_dim']))  # (B, 2H)
    return _bidirectional_lstm_cell(x, h0, c0, params['Ws'], params['Us'], params['Bs'])


def _bidirectional_gru_cell(x, h0, Ws, Us, Bs):
    '''
    Implements a bidirectional GRU cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        h0: Initial hidden state of shape (B, 2H), where:
            - H: Hidden state dimension
        Ws: Weight matrices for input-to-hidden transformations of shape (6, I, H).
            - First 3 matrices are for forward direction
            - Last 3 matrices are for backward direction
        Us: Weight matrices for hidden-to-hidden transformations of shape (6, H, H).
            - First 3 matrices are for forward direction
            - Last 3 matrices are for backward direction
        Bs: Bias vectors of shape (6, H).
            - First 3 vectors are for forward direction
            - Last 3 vectors are for backward direction

    Returns:
        res: Output sequence of shape (S, B, 2H).
        h: Final hidden state of shape (B, 2H).
    '''

    # Split parameters for forward and backward directions
    Ws_forward, Ws_backward = Ws[:3], Ws[3:]  # (3, I, H) each
    Us_forward, Us_backward = Us[:3], Us[3:]  # (3, H, H) each
    Bs_forward, Bs_backward = Bs[:3], Bs[3:]  # (3, H) each

    # Forward GRU
    res_forward, h_forward = _gru_cell(
        x,
        h0[:, :h0.shape[-1] // 2],  # Use first half of h0
        tuple(Ws_forward),
        tuple(Us_forward),
        tuple(Bs_forward)
    )

    # Backward GRU
    res_backward, h_backward = _gru_cell(
        jnp.flip(x, axis=0),
        h0[:, h0.shape[-1] // 2:],  # Use second half of h0
        tuple(Ws_backward),
        tuple(Us_backward),
        tuple(Bs_backward)
    )

    # Concatenate results
    return (
        jnp.concatenate([res_forward, jnp.flip(res_backward, axis=0)], axis=-1),
        jnp.concatenate([h_forward, h_backward], axis=-1)
    )


def get_bigru(timesteps, input_dim, hidden_dim, strategy='Xavier'):
    '''
    Returns a dictionary containing the hyperparameters of the bidirectional GRU cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.
        strategy: Initialize strategy, a str, including None, Kaiming, Xavier.

    Returns:
        A dictionary containing the hyperparameters of the bidirectional GRU cell.
    '''
    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'strategy': strategy,
    }


def bigru(x, params, config):
    '''
    Implements a bidirectional GRU cell using `lax.scan` for optimization.

    Args:
        x: Input sequence of shape (S, B, I), where:
           - S: Sequence length
           - B: Batch size
           - I: Input dimension
        params: Dictionary containing the parameters:
            - Ws: Weight matrices for input-to-hidden transformations of shape (6, I, H).
            - Us: Weight matrices for hidden-to-hidden transformations of shape (6, H, H).
            - Bs: Bias vectors of shape (6, H).
        config: Dictionary containing the hyperparameters.

    Returns:
        res: Output sequence of shape (S, B, 2H).
        h: Final hidden state of shape (B, 2H).
    '''
    h0 = jnp.zeros((x.shape[1], 2 * config['hidden_dim']))  # (B, 2H)
    return _bidirectional_gru_cell(x, h0, params['Ws'], params['Us'], params['Bs'])

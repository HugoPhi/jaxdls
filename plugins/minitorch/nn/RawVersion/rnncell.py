import jax.numpy as jnp
from ...utils import sigmoid


def _basic_rnn_cell(x, h0,
                    w_hh, w_xh, b_h,
                    w_hy, b_y):
    '''
    Implements a basic RNN cell using an explicit loop.

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
        h: Hidden state sequence of shape (S, B, H).
    '''

    steps, batch_size, input_dim = x.shape  # S, B, I
    _, hidden_dim = w_hh.shape  # H, H
    _, output_dim = w_hy.shape  # H, O

    res = jnp.zeros((steps, batch_size, output_dim))  # S, B, O
    h = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
    h = h.at[-1].set(h0)
    for ix in range(steps):
        h = h.at[ix].set(
            jnp.tanh(h[ix - 1] @ w_hh + x[ix] @ w_xh + b_h)
        )
        res = res.at[ix].set(
            h[ix] @ w_hy + b_y
        )

    return res, h


def get_basic_rnn(timesteps, input_dim, output_dim, hidden_dim):
    '''
    Returns a dictionary containing the hyperparameters of the basic RNN cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_dim: Hidden state dimension.

    Returns:
        A dictionary containing the hyperparameters of the basic RNN cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim
    }


def basic_rnn(x, params):
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
    Implements an LSTM cell using an explicit loop.

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
        h: Hidden state sequence of shape (S, B, H).
        c: Cell state sequence of shape (S, B, H).
    '''

    w_i, w_f, w_c, w_o = Ws  # (I, H)
    u_i, u_f, u_c, u_o = Us  # (H, H)
    b_i, b_f, b_c, b_o = Bs  # (H)

    steps, batch_size, input_dim = x.shape  # S, B, I
    _, hidden_dim = w_i.shape

    res = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
    h = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
    h = h.at[-1].set(h0)
    c = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
    c = c.at[-1].set(c0)

    for ix in range(steps):
        II = sigmoid(x[ix] @ w_i + h[ix - 1] @ u_i + b_i)
        FF = sigmoid(x[ix] @ w_f + h[ix - 1] @ u_f + b_f)
        CC = jnp.tanh(x[ix] @ w_c + h[ix - 1] @ u_c + b_c)
        OO = sigmoid(x[ix] @ w_o + h[ix - 1] @ u_o + b_o)

        c = c.at[ix].set(
            FF * c[ix - 1] + II * CC
        )
        h = h.at[ix].set(
            OO * jnp.tanh(CC)
        )
        res = res.at[ix].set(
            OO
        )

    return res, h, c


def get_lstm(timesteps, input_dim, hidden_dim):
    '''
    Returns a dictionary containing the hyperparameters of the LSTM cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.

    Returns:
        A dictionary containing the hyperparameters of the LSTM cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim
    }


def lstm(x, params, config):
    h0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    c0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    Ws = params['Ws']
    Us = params['Us']
    Bs = params['Bs']
    return _lstm_cell(x, h0, c0, Ws, Us, Bs)


def _gru_cell(x, h0,
              Ws, Us, Bs):
    '''
    Implements a GRU cell using an explicit loop.

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
        h: Hidden state sequence of shape (S, B, H).
    '''

    w_z, w_r, w_h = Ws  # (I, H)
    u_z, u_r, u_h = Us  # (H, H)
    b_z, b_r, b_h = Bs  # (H)

    steps, batch_size, input_dim = x.shape  # S, B, I
    _, hidden_dim = w_z.shape

    h = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
    h = h.at[-1].set(h0)

    for ix in range(steps):
        R = sigmoid(x[ix] @ w_r + h[ix - 1] @ u_r + b_r)
        Z = sigmoid(x[ix] @ w_z + h[ix - 1] @ u_z + b_z)

        H = jnp.tanh(x[ix] @ w_h + (R * h[ix - 1]) @ u_h + b_h)

        h = h.at[ix].set(
            (1 - Z) * h[ix - 1] + Z * H
        )

    return h


def get_gru(timesteps, input_dim, hidden_dim):
    '''
    Returns a dictionary containing the hyperparameters of the GRU cell.

    Args:
        timesteps: Number of timesteps.
        input_dim: Input dimension.
        hidden_dim: Hidden state dimension.

    Returns:
        A dictionary containing the hyperparameters of the GRU cell.
    '''

    return {
        'time_steps': timesteps,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim
    }


def gru(x, params, config):
    h0 = jnp.zeros((x.shape[1], config['hidden_dim']))
    Ws = params['Ws']
    Us = params['Us']
    Bs = params['Bs']
    return _gru_cell(x, h0, Ws, Us, Bs)

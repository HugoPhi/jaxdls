import jax.numpy as jnp
from jax import lax, jit


class RawVersion:
    @staticmethod
    def basic_rnn_cell(x, h0,
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

    @staticmethod
    def lstm_cell(x, h0, c0,
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

        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

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

    @staticmethod
    def gru_cell(x, h0,
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

        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

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


class JaxOptimized:
    @staticmethod
    def basic_rnn_cell(x, h0,
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

    @staticmethod
    def lstm_cell(x, h0, c0,
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

        def sigmoid(x):
            x = jnp.clip(x, -50, 50)
            return 1 / (1 + jnp.exp(-x))

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

    @staticmethod
    def gru_cell(x, h0,
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

        def sigmoid(x):
            x = jnp.clip(x, -50, 50)
            return 1 / (1 + jnp.exp(-x))

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

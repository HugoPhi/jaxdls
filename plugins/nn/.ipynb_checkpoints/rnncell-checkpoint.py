import jax.numpy as np
from jax import random, jit, random, lax

class RawVersion:
    @staticmethod
    def normal_cell(x, h0, 
                    w_hh, w_xh, b_h, 
                    w_hy, b_y):
        '''
        Input
        -----
        x: (S, B, I)
        h0: (B, H)
        q_hh: (H, H)
        w_xh: (I, H)
        b_h: (H)
        w_hy: (H, O)
        b_y: (H)

        Output
        ------
        res: (S, B, O)
        h: (S, B, H)
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
        Input
        -----
        x: (S, B, I)
        h0: (B, H)
        c0: (B, H)
        Ws: 4 * (I, H)
        Us: 4 * (H, H)
        Bs: 4 * (H)

        Output
        ------
        res: (S, B, H)
        h: (S, B, H)
        c: (S, B, H)
        '''

        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        w_i, w_f, w_c, w_o = Ws  # (I, H)
        u_i, u_f, u_c, u_o = Us  # (H, H)
        b_i, b_f, b_c, b_o = Bs  # (H)

        steps, batch_size, input_dim = x.shape  # S, B, I
        _, hidden_dim = w_i.shape

        res = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
        h = jnp.zeros((steps, batch_size, hidden_dim)) # S, B, H
        h = h.at[-1].set(h0)
        c = jnp.zeros((steps, batch_size, hidden_dim))  # S, B, H
        c = c.at[-1].set(c0)

        for ix in range(steps):
            I = sigmoid(x[ix] @ w_i + h[ix - 1] @ u_i + b_i)
            F = sigmoid(x[ix] @ w_f + h[ix - 1] @ u_f + b_f)
            C = jnp.tanh(x[ix] @ w_c + h[ix - 1] @ u_c + b_c)
            O = sigmoid(x[ix] @ w_o + h[ix - 1] @ u_o + b_o)

            c = c.at[ix].set(
                F*c[ix - 1] + I*C
            )
            h = h.at[ix].set(
                O*jnp.tanh(C)
            )
            res = res.at[ix].set(
                O
            )

        return res, h, c

    

    @staticmethod
    def gru_cell(x, h0, 
                    Ws, Us, Bs):
        '''
        Input
        -----
        x: (S, B, I)
        h0: (S, B, H)
        Ws: 3 * (I, H)
        Us: 3 * (H, H)
        Bs: 3 * (H)

        Output
        ------
        res: (S, B, H)
        h: (S, B, H)
        '''

        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        w_z, w_r, w_h = Ws  # (I, H)
        u_z, u_r, u_h = Us  # (H, H)
        b_z, b_r, b_h = Bs  # (H)

        steps, batch_size, input_dim = x.shape  # S, B, I
        _, hidden_dim = w_z.shape

        h = jnp.zeros((steps, batch_size, hidden_dim)) # S, B, H
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
    def normal_cell(x, h0, 
                    w_hh, w_xh, b_h, 
                    w_hy, b_y):
        '''
        Input
        -----
        x: (S, B, I)
        h0: (B, H)
        q_hh: (H, H)
        w_xh: (I, H)
        b_h: (H)
        w_hy: (H, O)
        b_y: (H)

        Output
        ------
        res: (S, B, O)
        h: (B, H)  # 最后一个h状态
        '''
        def step(carry, x_t):
            h_prev = carry[0]
            
            h_new = jnp.tanh(h_prev @ w_hh + x_t @ w_xh + b_h)
            res = h_new @ w_hy + b_y 

            return (h_new), (res)
            

        (h), (res) = lax.scan(step, (h0), x)

        return res, h
    
    @staticmethod
    def lstm_cell(x, h0, c0, 
                  Ws, Us, Bs):
        '''
        Input
        -----
        x: (S, B, I)
        h0: (B, H)
        c0: (B, H)
        Ws: 4 * (I, H)
        Us: 4 * (H, H)
        Bs: 4 * (H)

        Output
        ------
        res: (S, B, H)
        h: (B, H)  # 最后一个h状态
        c: (B, H)  # 最后一个c状态
        '''
        
        def sigmoid(x):
            x = jnp.clip(x, -50, 50)
            return 1 / (1 + jnp.exp(-x))
    
        w_i, w_f, w_c, w_o = Ws  # (I, H)
        u_i, u_f, u_c, u_o = Us  # (H, H)
        b_i, b_f, b_c, b_o = Bs  # (H)

        def step(carry, x_t):
            h_prev, c_prev = carry
            I = sigmoid(x_t @ w_i + h_prev @ u_i + b_i)
            F = sigmoid(x_t @ w_f + h_prev @ u_f + b_f)
            C = jnp.tanh(x_t @ w_c + h_prev @ u_c + b_c)
            O = sigmoid(x_t @ w_o + h_prev @ u_o + b_o)
    
            c_new = F * c_prev + I * C
            h_new = O * jnp.tanh(c_new)
            res_new = O
    
            return (h_new, c_new), (res_new)
    
        (h, c), (res) = lax.scan(step, (h0, c0), x)  # use scan to decrease RAM usage, I do not know why old version ram will increse by epochs
        return res, h, c

    @staticmethod
    def gru_cell(x, h0, 
                    Ws, Us, Bs):
        '''
        Input
        -----
        x: (S, B, I)
        h0: (S, B, H)
        Ws: 3 * (I, H)
        Us: 3 * (H, H)
        Bs: 3 * (H)

        Output
        ------
        res: (S, B, H)
        h: (B, H)
        '''

        def sigmoid(x):
            x = jnp.clip(x, -50, 50)
            return 1 / (1 + jnp.exp(-x))

        w_z, w_r, w_h = Ws  # (I, H)
        u_z, u_r, u_h = Us  # (H, H)
        b_z, b_r, b_h = Bs  # (H)

        def step(carry, x_t):
            h_prev = carry[0]

            R = sigmoid(x_t @ w_r + h_prev @ u_r + b_r)
            Z = sigmoid(x_t @ w_z + h_prev @ u_z + b_z)
            
            H = jnp.tanh(x_t @ w_h + (R * h_prev) @ u_h + b_h)

            new_h = (1 - Z) @ h_prev + Z * H

            return (new_h), (new_h)

        (h), (res) = lax.scan(step, (h0), x)

        return h, res

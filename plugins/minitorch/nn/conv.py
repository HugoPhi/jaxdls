import jax.numpy as jnp
from jax import lax


class RawVersion:
    @staticmethod
    def conv2d(x, w, b, padding=1):
        '''
        Performs a 2D convolution operation using explicit loops.

        Args:
            x: Input tensor of shape (B, I, H, W), where:
               - B: Batch size
               - I: Input channels
               - H: Height
               - W: Width
            w: Convolution kernel of shape (O, I, KH, KW), where:
               - O: Output channels
               - KH: Kernel height
               - KW: Kernel width
            b: Bias term of shape (O,).
            padding: Padding size (default: 1).

        Returns:
            fgraph: Output feature map of shape (B, O, H', W'), where:
                    - H': Output height = H + 2 * padding - KH + 1
                    - W': Output width = W + 2 * padding - KW + 1
        '''

        bs, icl, he, wi = x.shape  # input graph -> batch_size x channel x height x width
        ocl, icl, kh, kw = w.shape
        he = (he + 2 * padding - kh + 1)
        wi = (wi + 2 * padding - kw + 1)

        fgraph = jnp.zeros((bs, ocl, he, wi))  # feature graph

        # padding for x
        pad_mat = (
            (0, 0),
            (0, 0),
            (padding, padding),
            (padding, padding)
        )

        x_padded = jnp.pad(x, pad_mat, mode='constant', constant_values=0)

        for k in range(ocl):
            for i in range(he):
                for j in range(wi):
                    fgraph.at[:, k, i, j].set(
                        jnp.sum(x_padded[:, :, i:i + kh, j:j + kw] * w[k], axis=(1, 2, 3)) + b[k]
                    )

        return fgraph

    @staticmethod
    def max_pooling2d(x, pool_size=(2, 2), stride=None):
        '''
        Performs 2D max pooling using explicit loops.

        Args:
            x: Input tensor of shape (B, C, H, W), where:
               - B: Batch size
               - C: Channels
               - H: Height
               - W: Width
            pool_size: Pooling window size (default: (2, 2)).
            stride: Stride for the pooling operation (default: None, same as pool_size).

        Returns:
            output_array: Output tensor of shape (B, C, H', W'), where:
                          - H': Output height = (H - pool_height) // stride_height + 1
                          - W': Output width = (W - pool_width) // stride_width + 1
        '''

        if stride is None:
            stride = pool_size

        batch_size, channels, height, width = x.shape
        pool_height, pool_width = pool_size
        stride_height, stride_width = stride

        output_height = (height - pool_height) // stride_height + 1
        output_width = (width - pool_width) // stride_width + 1

        output_array = jnp.zeros((batch_size, channels, output_height, output_width))

        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        window = x[n, c,
                                   i * stride_height:i * stride_height + pool_height,
                                   j * stride_width:j * stride_width + pool_width]
                        output_array.at[n, c, i, j].set(
                            jnp.max(window)
                        )

        return output_array

    @staticmethod
    def conv1d(x, w, b, padding=1):
        '''
        Performs a 1D convolution operation using explicit loops.

        Args:
            x: Input tensor of shape (B, I, L), where:
               - B: Batch size
               - I: Input channels
               - L: Length
            w: Convolution kernel of shape (O, I, KL), where:
               - O: Output channels
               - KL: Kernel length
            b: Bias term of shape (O,).
            padding: Padding size (default: 1).

        Returns:
            fgraph: Output feature map of shape (B, O, L'), where:
                    - L': Output length = L + 2 * padding - KL + 1
        '''

        bs, icl, le = x.shape  # input shape: (B, C, L)
        ocl, icl, kl = w.shape
        output_length = le + 2 * padding - kl + 1

        fgraph = jnp.zeros((bs, ocl, output_length))
        pad_mat = ((0, 0), (0, 0), (padding, padding))
        x_padded = jnp.pad(x, pad_mat, mode='constant')

        for k in range(ocl):
            for j in range(output_length):
                window = x_padded[:, :, j:j + kl]
                fgraph.at[:, k, j].set(
                    jnp.sum(window * w[k], axis=(1, 2)) + b[k]
                )
        return fgraph

    @staticmethod
    def conv3d(x, w, b, padding=1):
        '''
        Performs a 3D convolution operation using explicit loops.

        Args:
            x: Input tensor of shape (B, I, D, H, W), where:
               - B: Batch size
               - I: Input channels
               - D: Depth
               - H: Height
               - W: Width
            w: Convolution kernel of shape (O, I, KD, KH, KW), where:
               - O: Output channels
               - KD: Kernel depth
               - KH: Kernel height
               - KW: Kernel width
            b: Bias term of shape (O,).
            padding: Padding size (default: 1).

        Returns:
            fgraph: Output feature map of shape (B, O, D', H', W'), where:
                    - D': Output depth = D + 2 * padding - KD + 1
                    - H': Output height = H + 2 * padding - KH + 1
                    - W': Output width = W + 2 * padding - KW + 1
        '''

        bs, icl, d, h, w_dim = x.shape
        ocl, icl, kd, kh, kw = w.shape
        output_d = d + 2 * padding - kd + 1
        output_h = h + 2 * padding - kh + 1
        output_w = w_dim + 2 * padding - kw + 1

        fgraph = jnp.zeros((bs, ocl, output_d, output_h, output_w))
        pad_mat = ((0, 0), (0, 0), (padding, padding),
                   (padding, padding), (padding, padding))
        x_padded = jnp.pad(x, pad_mat, mode='constant')

        for k in range(ocl):
            for i in range(output_d):
                for j in range(output_h):
                    for le in range(output_w):
                        window = x_padded[:, :, i:i + kd, j:j + kh, le:le + kw]
                        fgraph.at[:, k, i, j, le].set(
                            jnp.sum(window * w[k], axis=(1, 2, 3, 4)) + b[k]
                        )
        return fgraph

    @staticmethod
    def max_pooling1d(x, pool_size=2, stride=None):
        '''
        Performs 1D max pooling using explicit loops.

        Args:
            x: Input tensor of shape (B, C, L), where:
               - B: Batch size
               - C: Channels
               - L: Length
            pool_size: Pooling window size (default: 2).
            stride: Stride for the pooling operation (default: None, same as pool_size).

        Returns:
            out: Output tensor of shape (B, C, L'), where:
                 - L': Output length = (L - pool_size) // stride + 1
        '''

        stride = stride or pool_size
        batch, ch, le = x.shape
        output_l = (le - pool_size) // stride + 1

        out = jnp.zeros((batch, ch, output_l))
        for n in range(batch):
            for c in range(ch):
                for i in range(output_l):
                    start = i * stride
                    out.at[n, c, i].set(jnp.max(x[n, c, start:start + pool_size]))
        return out

    @staticmethod
    def max_pooling3d(x, pool_size=(2, 2, 2), stride=None):
        '''
        Performs 3D max pooling using explicit loops.

        Args:
            x: Input tensor of shape (B, C, D, H, W), where:
               - B: Batch size
               - C: Channels
               - D: Depth
               - H: Height
               - W: Width
            pool_size: Pooling window size (default: (2, 2, 2)).
            stride: Stride for the pooling operation (default: None, same as pool_size).

        Returns:
            out: Output tensor of shape (B, C, D', H', W'), where:
                 - D': Output depth = (D - pool_d) // stride_d + 1
                 - H': Output height = (H - pool_h) // stride_h + 1
                 - W': Output width = (W - pool_w) // stride_w + 1
        '''

        stride = stride or pool_size
        batch, ch, d, h, w = x.shape
        pool_d, pool_h, pool_w = pool_size
        stride_d, stride_h, stride_w = stride
        out_d = (d - pool_d) // stride_d + 1
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1

        out = jnp.zeros((batch, ch, out_d, out_h, out_w))
        for n in range(batch):
            for c in range(ch):
                for i in range(out_d):
                    for j in range(out_h):
                        for le in range(out_w):
                            di = i * stride_d
                            dj = j * stride_h
                            dl = le * stride_w
                            window = x[n, c, di:di + pool_d, dj:dj + pool_h, dl:dl + pool_w]
                            out.at[n, c, i, j, le].set(jnp.max(window))
        return out


class JaxOptimized:
    @staticmethod
    def conv2d(x, w, b, padding=1):
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

    @staticmethod
    def max_pooling2d(x, pool_size=(2, 2), stride=None):
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

    @staticmethod
    def conv1d(x, w, b, padding=1):
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

    @staticmethod
    def conv3d(x, w, b, padding=1):
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

    @staticmethod
    def max_pooling1d(x, pool_size=2, stride=None):
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

    @staticmethod
    def max_pooling3d(x, pool_size=(2, 2, 2), stride=None):
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

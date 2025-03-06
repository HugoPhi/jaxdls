import jax.numpy as jnp
from jax import lax


class RawVersion:
    @staticmethod
    def conv2d(x, w, b, padding=1):
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


class JaxOptimized:
    @staticmethod
    def conv2d(x, w, b, padding=1):
        '''
        Input
        -----
        x: (B, I, H, W)
        w: (O, I, KH, KW)
        b: (O)

        Output
        ------
        res: (B, O, H, W)
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
        Input
        -----
        x: (B, C, H, W)

        Output
        ------
        res: (B, C, H, W)
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

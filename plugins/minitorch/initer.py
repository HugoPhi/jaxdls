import jax.numpy as jnp
from jax import random


class Initer:
    '''
    A class for initializing parameters of various neural network layers using appropriate initialization schemes.
    Filter out static parameters & return trainable parameters

    Supported layer types:
    - Long Short-Term Memory (lstm)
    - Gated Recurrent Unit (gru)
    - Fully Connected (fc) layers
    - Fully Connected layers for ReLU activation (fc4relu)
    - 2D Convolutional layers (conv2d)
    - 2D Convolutional layers for ReLU activation (conv2d4relu)
    - Dropout layer

    Their name should be like:
    - "lstm:"
    - "gru:"
    - "fc:"
    - "fc4relu:"
    - "conv2d:"
    - "conv2d4relu:"
    - "dropout:"

    Attributes:
        key: A JAX random key for generating random values.
        config: A dictionary containing configuration details for each layer.
    '''

    # TODO: conv1d, conv3d
    SupportLayers = ('lstm', 'gru',
                     'fc', 'fc4relu',
                     'conv2d', 'conv2d4relu', 'conv1d', 'conv1d4relu', 'conv3d', 'conv3d4relu')

    def __init__(self, config, key):
        '''
        Initializes the Initer class.

        Args:
            config: A dictionary containing configuration details for each layer.
                    Keys should be in the format `layer_type:layer_name`, and values
                    should be dictionaries specifying the required dimensions.
            key: A JAX random key for generating random values.
        '''

        self.key = key
        self.config = {k: v for k, v in config.items() if k.split(':')[0] in Initer.SupportLayers}  # filter out key not in SupportLayers

    def __call__(self):
        '''
        Initializes parameters for all layers specified in the configuration.

        Returns:
            A dictionary where keys are layer names and values are dictionaries of initialized parameters.
        '''

        return {k: self._init_param(k) for k in self.config.keys()}

    def _init_param(self, name: str):
        '''
        Initializes parameters for a specific layer.

        Args:
            name: The name of the layer in the format `layer_type:layer_name`.

        Returns:
            A dictionary of initialized parameters for the specified layer.

        Raises:
            ValueError: If the layer type is not supported.
        '''

        layer_type = name.split(':')[0]

        if layer_type not in Initer.SupportLayers:
            raise ValueError(f'[x] Do not support layer type: {layer_type} given by {name}.')

        f = getattr(self, f'_{layer_type}', None)

        return f(name)

    def _lstm(self, name):
        '''
        Initializes parameters for an LSTM layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (shape: (4, input_dim, hidden_dim)).
            - 'Us': Weight matrix for hidden-to-hidden transformations (shape: (4, hidden_dim, hidden_dim)).
            - 'Bs': Bias terms (shape: (4, hidden_dim)), with the forget gate bias initialized to 1.
        '''

        return {
            'Ws': random.normal(self.key, (
                4,
                self.config[name]['input_dim'],
                self.config[name]['hidden_dim'],
            )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),  # Xavier
            'Us': random.normal(self.key, (
                4,
                self.config[name]['hidden_dim'],
                self.config[name]['hidden_dim'],
            )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),
            'Bs': jnp.zeros((
                4,
                self.config[name]['hidden_dim']
            )).at[0].set(1),  # suggestion by Qwen.
        }

    def _gru(self, name):
        '''
        Initializes parameters for a GRU layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (shape: (3, input_dim, hidden_dim)).
            - 'Us': Weight matrix for hidden-to-hidden transformations (shape: (3, hidden_dim, hidden_dim)).
            - 'Bs': Bias terms (shape: (3, hidden_dim)).
        '''

        return {
            'Ws': random.normal(self.key, (
                3,
                self.config[name]['input_dim'],
                self.config[name]['hidden_dim'],
            )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),  # Xavier
            'Us': random.normal(self.key, (
                3,
                self.config[name]['hidden_dim'],
                self.config[name]['hidden_dim'],
            )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),
            'Bs': jnp.zeros((
                3,
                self.config[name]['hidden_dim']
            )),
        }

    def _fc(self, name):
        '''
        Initializes parameters for a fully connected (FC) layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'output_dim': int,  # Output dimension
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight matrix (shape: (input_dim, output_dim)).
            - 'b': Bias vector (shape: (output_dim,)).
        '''

        res = {
            'w': random.normal(self.key, (
                self.config[name]['input_dim'],
                self.config[name]['output_dim'],
            )),
            'b': jnp.zeros(
                self.config[name]['output_dim'],
            )
        }

        return res

    def _fc4relu(self, name):
        '''
        Initializes parameters for a fully connected (FC) layer intended for use with ReLU activation.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'output_dim': int,  # Output dimension
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight matrix (shape: (input_dim, output_dim)), initialized using Kaiming initialization.
            - 'b': Bias vector (shape: (output_dim,)).
        '''

        return {
            'w': random.normal(self.key, (
                self.config[name]['input_dim'],
                self.config[name]['output_dim'],
            )) * jnp.sqrt(2 / self.config[name]['input_dim']),  # Kaiming init
            'b': jnp.zeros(
                self.config[name]['output_dim'],
            )
        }

    def _conv2d(self, name):
        '''
        Initializes parameters for a 2D convolutional layer.

        Config should be:
        ```
        name: {
            'input_channel': int,  # Number of input channels
            'output_channel': int,  # Number of output channels
            'kernel_size': int,     # Size of the convolutional kernel
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight tensor (shape: (output_channel, input_channel, kernel_size, kernel_size)).
            - 'b': Bias vector (shape: (output_channel,)).
        '''

        return {
            'w': random.normal(self.key, (
                self.config[name]['output_channel'],
                self.config[name]['input_channel'],
                self.config[name]['kernel_size'],
                self.config[name]['kernel_size'],
            )),
            'b': jnp.zeros((self.config[name]['output_channel']))
        }

    def _conv2d4relu(self, name):
        '''
        Initializes parameters for a 2D convolutional layer intended for use with ReLU activation.

        Config should be:
        ```
        name: {
            'input_channel': int,  # Number of input channels
            'output_channel': int,  # Number of output channels
            'kernel_size': int,     # Size of the convolutional kernel
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight tensor (shape: (output_channel, input_channel, kernel_size, kernel_size)),
                     initialized using Kaiming initialization.
            - 'b': Bias vector (shape: (output_channel,)).
        '''

        return {
            'w': random.normal(self.key, (
                self.config[name]['output_channel'],
                self.config[name]['input_channel'],
                self.config[name]['kernel_size'],
                self.config[name]['kernel_size'],
            )) * jnp.sqrt(2 / (self.config[name]['output_channel'] * self.config[name]['input_channel'] * self.config[name]['kernel_size'])),
            'b': jnp.zeros((self.config[name]['output_channel']))
        }

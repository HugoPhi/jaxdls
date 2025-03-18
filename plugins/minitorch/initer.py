import jax.numpy as jnp
from jax import random


class Initer:
    '''
    A class for initializing parameters of various neural network layers using appropriate initialization schemes.
    Filter out static parameters & return trainable parameters

    Supported layer types:
    - Basic Recurrent Neural Network (basic_rnn)
    - Long Short-Term Memory (lstm)
    - Bidirectional Long Short-Term Memory (bilstm)
    - Gated Recurrent Unit (gru)
    - Bidirectional Gated Recurrent Unit (bigru)
    - Fully Connected (fc) layers
    - 1D Convolutional layers (conv1d)
    - 2D Convolutional layers (conv2d)
    - 3D Convolutional layers (conv3d)

    Their name should be like:
    - "basic_rnn:"
    - "lstm:"
    - "bilstm:"
    - "gru:"
    - "bigru:"
    - "fc:"
    - "conv1d:"
    - "conv2d:"
    - "conv3d:"

    Attributes:
        key: A JAX random key for generating random values.
        config: A dictionary containing configuration details for each layer.
    '''

    SupportLayers = ('basic_rnn', 'lstm', 'gru',
                     'bilstm', 'bigru',
                     'fc',
                     'conv1d', 'conv2d', 'conv3d')

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

    def _bilstm(self, name):
        '''
        Initializes parameters for a bidirectional LSTM layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (8, input_dim, hidden_dim).
                - First 4 matrices are for forward direction.
                - Last 4 matrices are for backward direction.
            - 'Us': Weight matrix for hidden-to-hidden transformations (8, hidden_dim, hidden_dim).
                - First 4 matrices are for forward direction.
                - Last 4 matrices are for backward direction.
            - 'Bs': Bias terms (8, hidden_dim).
                - First 4 biases are for forward direction.
                - Last 4 biases are for backward direction.
                - The forget gate bias (index 0 and 4) is initialized to 1.
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'Ws': random.normal(self.key, (
                        8,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Us': random.normal(self.key, (
                        8,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Bs': jnp.zeros((
                        8,
                        self.config[name]['hidden_dim']
                    )).at[0].set(1).at[4].set(1),  # Initialize forget gate biases to 1
                }
            case 'Kaiming':
                return {
                    'Ws': random.normal(self.key, (
                        8,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),  # Kaiming
                    'Us': random.normal(self.key, (
                        8,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),
                    'Bs': jnp.zeros((
                        8,
                        self.config[name]['hidden_dim']
                    )).at[0].set(1).at[4].set(1),  # Initialize forget gate biases to 1
                }
            case 'Xavier':
                return {
                    'Ws': random.normal(self.key, (
                        8,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),  # Xavier
                    'Us': random.normal(self.key, (
                        8,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),
                    'Bs': jnp.zeros((
                        8,
                        self.config[name]['hidden_dim']
                    )).at[0].set(1).at[4].set(1),  # Initialize forget gate biases to 1
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _bigru(self, name):
        '''
        Initializes parameters for a bidirectional GRU layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (6, input_dim, hidden_dim).
                - First 3 matrices are for forward direction.
                - Last 3 matrices are for backward direction.
            - 'Us': Weight matrix for hidden-to-hidden transformations (6, hidden_dim, hidden_dim).
                - First 3 matrices are for forward direction.
                - Last 3 matrices are for backward direction.
            - 'Bs': Bias terms (6, hidden_dim).
                - First 3 biases are for forward direction.
                - Last 3 biases are for backward direction.
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'Ws': random.normal(self.key, (
                        6,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Us': random.normal(self.key, (
                        6,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Bs': jnp.zeros((
                        6,
                        self.config[name]['hidden_dim']
                    )),
                }
            case 'Kaiming':
                return {
                    'Ws': random.normal(self.key, (
                        6,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),  # Kaiming
                    'Us': random.normal(self.key, (
                        6,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),
                    'Bs': jnp.zeros((
                        6,
                        self.config[name]['hidden_dim']
                    )),
                }
            case 'Xavier':
                return {
                    'Ws': random.normal(self.key, (
                        6,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),  # Xavier
                    'Us': random.normal(self.key, (
                        6,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),
                    'Bs': jnp.zeros((
                        6,
                        self.config[name]['hidden_dim']
                    )),
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _basic_rnn(self, name):
        '''
        Initializes parameters for a basic RNN layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'h0': Initial hidden state (hidden_dim,).
            - 'w_hh': Weight matrix for hidden-to-hidden transformations (hidden_dim, hidden_dim).
            - 'w_hx': Weight matrix for input-to-hidden transformations (input_dim, hidden_dim).
            - 'b_h': Bias term for hidden state (hidden_dim,).
            - 'w_hy': Weight matrix for hidden-to-output transformations (hidden_dim, output_dim).
            - 'b_y': Bias term for output (output_dim,).
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'h0': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])),
                    'w_hh': random.normal(self.key, (self.config[name]['hidden_dim'], self.config[name]['hidden_dim'])),
                    'w_hx': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])),
                    'b_h': jnp.zeros((self.config[name]['hidden_dim'])),
                    'w_hy': jnp.zeros((self.config[name]['hidden_dim'])),
                    'b_y': jnp.zeros((self.config[name]['output_dim'])),
                }
            case 'Kaiming':
                return {
                    'h0': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(2 / self.config[name]['input_dim']),
                    'w_hh': random.normal(self.key, (self.config[name]['hidden_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(2 / self.config[name]['hidden_dim']),
                    'w_hx': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(2 / self.config[name]['input_dim']),
                    'b_h': jnp.zeros((self.config[name]['hidden_dim'])),
                    'w_hy': jnp.zeros((self.config[name]['hidden_dim'])),
                    'b_y': jnp.zeros((self.config[name]['output_dim'])),
                }
            case 'Xavier':
                return {
                    'h0': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(2 / (self.config[name]['input_dim'] + self.config[name]['hidden_dim'])),
                    'w_hh': random.normal(self.key, (self.config[name]['hidden_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(1 / self.config[name]['hidden_dim']),
                    'w_hx': random.normal(self.key, (self.config[name]['input_dim'], self.config[name]['hidden_dim'])) * jnp.sqrt(2 / self.config[name]['input_dim'] + self.config[name]['hidden_dim']),
                    'b_h': jnp.zeros((self.config[name]['hidden_dim'])),
                    'w_hy': jnp.zeros((self.config[name]['hidden_dim'])),
                    'b_y': jnp.zeros((self.config[name]['output_dim'])),
                }

    def _lstm(self, name):
        '''
        Initializes parameters for an LSTM layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (4, input_dim, hidden_dim).
            - 'Us': Weight matrix for hidden-to-hidden transformations (4, hidden_dim, hidden_dim).
            - 'Bs': Bias terms (4, hidden_dim), with the forget gate bias initialized to 1.
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'Ws': random.normal(self.key, (
                        4,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Us': random.normal(self.key, (
                        4,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Bs': jnp.zeros((
                        4,
                        self.config[name]['hidden_dim']
                    )),
                }
            case 'Kaiming':
                return {
                    'Ws': random.normal(self.key, (
                        4,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),  # Kaiming
                    'Us': random.normal(self.key, (
                        4,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),
                    'Bs': jnp.zeros((
                        4,
                        self.config[name]['hidden_dim']
                    )).at[0].set(1),  # suggestion by Qwen.
                }
            case 'Xavier':
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
                    )).at[0].set(1),
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _gru(self, name):
        '''
        Initializes parameters for a GRU layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'hidden_dim': int,  # Hidden state dimension
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'Ws': Weight matrix for input-to-hidden transformations (3, input_dim, hidden_dim).
            - 'Us': Weight matrix for hidden-to-hidden transformations (3, hidden_dim, hidden_dim).
            - 'Bs': Bias terms (3, hidden_dim).
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'Ws': random.normal(self.key, (
                        3,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Us': random.normal(self.key, (
                        3,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )),
                    'Bs': jnp.zeros((
                        3,
                        self.config[name]['hidden_dim']
                    ))
                }
            case 'Kaiming':
                return {
                    'Ws': random.normal(self.key, (
                        3,
                        self.config[name]['input_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),  # Kaiming
                    'Us': random.normal(self.key, (
                        3,
                        self.config[name]['hidden_dim'],
                        self.config[name]['hidden_dim'],
                    )) * jnp.sqrt(2 / (self.config[name]['input_dim'])),
                    'Bs': jnp.zeros((
                        3,
                        self.config[name]['hidden_dim']
                    )),
                }
            case 'Xavier':
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
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _fc(self, name):
        '''
        Initializes parameters for a fully connected (FC) layer.

        Config should be:
        ```
        name: {
            'input_dim': int,  # Input dimension
            'output_dim': int,  # Output dimension
            'starategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight matrix (input_dim, output_dim).
            - 'b': Bias vector (output_dim,).
        '''

        match self.config[name]['strategy']:
            case 'None':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['input_dim'],
                        self.config[name]['output_dim'],
                    )),
                    'b': jnp.zeros(
                        self.config[name]['output_dim'],
                    )
                }
            case 'Kaiming':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['input_dim'],
                        self.config[name]['output_dim'],
                    )) * jnp.sqrt(2 / self.config[name]['input_dim']),  # Kaiming init
                    'b': jnp.zeros(
                        self.config[name]['output_dim'],
                    )
                }
            case 'Xavier':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['input_dim'],
                        self.config[name]['output_dim'],
                    )) * jnp.sqrt(1 / (self.config[name]['input_dim'])),  # Xavier
                    'b': jnp.zeros(
                        self.config[name]['output_dim'],
                    )
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _conv2d(self, name):
        '''
        Initializes parameters for a 2D convolutional layer.

        Config should be:
        ```
        name: {
            'input_channel': int,  # Number of input channels
            'output_channel': int,  # Number of output channels
            'kernel_size': (int, int),     # Size of the convolutional kernel
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight tensor (output_channel, input_channel, *kernel_size).
            - 'b': Bias vector (output_channel,).
        '''

        match self.config[name]['strategy']:
            case 'Kaiming':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                    )) * jnp.sqrt(2 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0] * self.config[name]['kernel_size'][1])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'Xavier':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                    )) * jnp.sqrt(1 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0] * self.config[name]['kernel_size'][1])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'None':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                    )),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _conv1d(self, name):
        '''
        Initializes parameters for a 1D convolutional layer.

        Config should be:
        ```
        name: {
            'input_channel': int,  # Number of input channels
            'output_channel': int,  # Number of output channels
            'kernel_size': (int,),     # Size of the convolutional kernel
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight tensor (output_channel, input_channel, *kernel_size).
            - 'b': Bias vector (output_channel,).
        '''

        match self.config[name]['strategy']:
            case 'Kaiming':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                    )) * jnp.sqrt(2 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'Xavier':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                    )) * jnp.sqrt(1 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'None':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                    )),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

    def _conv3d(self, name):
        '''
        Initializes parameters for a 3D convolutional layer.

        Config should be:
        ```
        name: {
            'input_channel': int,  # Number of input channels
            'output_channel': int,  # Number of output channels
            'kernel_size': (int, int, int),     # Size of the convolutional kernel
            'strategy': str,  # Initial strategy, including None, Kaiming, Xavier
        }
        ```

        Returns:
            A dictionary containing:
            - 'w': Weight tensor (output_channel, input_channel, *kernel_size).
            - 'b': Bias vector (output_channel,).
        '''

        match self.config[name]['strategy']:
            case 'Kaiming':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                        self.config[name]['kernel_size'][2],
                    )) * jnp.sqrt(2 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0] * self.config[name]['kernel_size'][1] * self.config[name]['kernel_size'][2])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'Xavier':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                        self.config[name]['kernel_size'][2],
                    )) * jnp.sqrt(1 / (self.config[name]['input_channel'] * self.config[name]['kernel_size'][0] * self.config[name]['kernel_size'][1] * self.config[name]['kernel_size'][2])),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case 'None':
                return {
                    'w': random.normal(self.key, (
                        self.config[name]['output_channel'],
                        self.config[name]['input_channel'],
                        self.config[name]['kernel_size'][0],
                        self.config[name]['kernel_size'][1],
                        self.config[name]['kernel_size'][2],
                    )),
                    'b': jnp.zeros((self.config[name]['output_channel']))
                }
            case _:
                raise ValueError(f'[x] Do not support strategy: {name["strategy"]} given by {name}.')

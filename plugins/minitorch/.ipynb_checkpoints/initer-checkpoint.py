import jax.numpy as jnp
from jax import random


class Initer:
    '''
    supported:
    "lstm:"
    "fc:"
    "fc4relu:"
    "conv2d:"
    "conv2d4relu:"
    '''

    SupportLayers = ('lstm',
                     'fc', 'fc4relu',
                     'conv2d', 'conv2d4relu')

    def __init__(self, config, key):
        self.key = key
        self.config = {k: v for k, v in config.items() if k.split(':')[0] in Initer.SupportLayers}  # filter out key not in SupportLayers

    def __call__(self):
        return {k: self._init_param(k) for k in self.config.keys()}

    def _init_param(self, name: str):
        layer_type = name.split(':')[0]

        if layer_type not in Initer.SupportLayers:
            raise ValueError(f'[x] Do not support layer type: {layer_type} given by {name}.')

        f = getattr(self, f'_{layer_type}', None)

        return f(name)

    def _lstm(self, name):
        '''
        Config should be:

        ```
        name: {
            'input_dim': _,
            'hidden_dim': _,
        }
        ```
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

    def _fc(self, name):
        '''
        Config should be:

        ```
        name: {
            'input_dim': _,
            'output_dim': _,
        }
        ```
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
        Config should be:

        ```
        name: {
            'input_dim': _,
            'output_dim': _,
        }
        ```
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
        Config should be:

        ```
        name: {
            'input_channel': _,
            'output_channel': _,
            'kernel_size': _,
        }
        ```
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
        Config should be:

        ```
        name: {
            'input_channel': _,
            'output_channel': _,
            'kernel_size': _,
        }
        ```
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

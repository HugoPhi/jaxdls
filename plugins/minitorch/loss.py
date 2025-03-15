from abc import ABC, abstractmethod
from .utils import cross_entropy_loss
import jax.numpy as jnp
from jax import jit


class Loss(ABC):

    def __init__(self, f):
        '''
        f: x, params, train -> y_proba
        '''
        self.f = f

    @abstractmethod
    def get_loss(self, train):
        '''
        loss function: params, x, y_true -> loss
        '''
        pass

    @abstractmethod
    def get_embed_loss(self, x, y_true, train):
        '''
        embed loss funtion: params -> loss
        '''
        pass


class CrossEntropyLoss(Loss):

    def __init__(self, f):
        super().__init__(f)

    def get_loss(self, train):
        loss_function = lambda params, x, y_true: cross_entropy_loss(y_true, self.f(x, params, train))
        return loss_function

    def get_embed_loss(self, x, y_true, train):
        embed_loss_function = lambda params: cross_entropy_loss(y_true, self.f(x, params, train))
        return embed_loss_function


class BasicVAELoss(Loss):
    def __init__(self, f):
        super(BasicVAELoss, self).__init__(f)

    def get_loss(self):
        def _loss(params, x, y, key, epsilon=1e-9):
            x_recon, mu, logvar = self.f(params, x, key)
            x_recon_clip = jnp.clip(x_recon, epsilon, 1. - epsilon)  # clip here is very important, or you will get Nan when you training.

            bce = -jnp.sum(y * jnp.log(x_recon_clip) + (1 - y) * jnp.log(1 - x_recon_clip), axis=1)
            kld = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)

            return jnp.mean(bce + kld)

        return jit(_loss)

    def get_embed_loss(self):
        raise ValueError('this method is not accessable for BCE_KLD')

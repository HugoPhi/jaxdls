import jax.numpy as jnp
from jax import grad, vmap, jit, random, tree
from abc import ABC, abstractmethod

def softmax(logits):
    logits_stable = logits - jnp.max(logits, axis=1, keepdims=True)
    exp_logits = jnp.exp(logits_stable)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(y, y_pred):
    epsilon = 1e-9
    y_pred_clipped = jnp.clip(y_pred, epsilon, 1. - epsilon)  # clip here is very important, or you will get Nan when you training. 
    loss = -jnp.sum(y * jnp.log(y_pred_clipped), axis=1)
    return loss.mean()

class Optimizter(ABC):
    '''
    A Class included by model; regarded as a container for weights & update weights by steps.
    '''
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def flash(self):
        pass
        
    def open(self, _loss):
        '''
        Input
        -----
        x_tarin: training set input
        y_train: training set label
        '''

        if self.open is True:
            print('oprimizer is already opened.')
        else:
            self.flash()
            self._loss = _loss
            self.open = True

    def close(self):
        if self.open is False:
            print('oprimizer is already closed.')
        else: 
            self.open = False

    def get_params(self):
        if self.open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            return self.params

    def get_steps(self):
        if self.open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            return self.steps


class Adam(Optimizter):
    def __init__(self, params, 
                 lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-6):
        super().__init__()
        
        self.params = params
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def flash(self):
        self.V = tree.map(lambda x: jnp.zeros_like(x), self.params)
        self.VV = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0
        self.open = False

   
    def update(self):
        if self.open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            d_params = grad(self._loss, argnums=0)(self.params)
    
            t = self.steps + 1

            @jit
            def adam(d_w, w, v, vv):
                new_v = self.beta1*v + (1 - self.beta1)*d_w
                new_vv = self.beta2*vv + (1 - self.beta2)*d_w*d_w
    
                v_hat = new_v / (1 - self.beta1**t)
                vv_hat = new_vv / (1 - self.beta2**t)
                step = - self.lr * v_hat / (jnp.sqrt(vv_hat) + self.epsilon)
    
                new_w = w + step
                return jnp.stack((
                    new_w,
                    new_v,
                    new_vv,
                ))
    
            def decode(pack, num_return=3):
                res = []
                for i in range(num_return):
                    res.append(tree.map(lambda x: x[i], pack))

                return res
    
            pack = tree.map(adam, d_params, self.params, self.V, self.VV)
            self.params, self.V, self.VV = decode(pack)
            self.steps += 1



class RawGD(Optimizter):
    def __init__(self, params, 
                 lr=0.01):
        super().__init__()
        
        self.params = params
        self.lr = lr

    def flash(self):
        self.steps = 0
        self.open = False

    def update(self):
        if self.open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            d_params = grad(self._loss, argnums=0)(self.params)

            @jit
            def gd(d_w, w):
                new_w = w - self.lr * d_w
                return new_w

            self.params = tree.map(gd, d_params, self.params)
            self.steps += 1
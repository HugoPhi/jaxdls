import jax.numpy as jnp
from ..utils import cross_entropy_loss
from jax import jit
from abc import ABC, abstractmethod
from .conv import JaxOptimized as Conv
from .rnncell import JaxOptimized as Rnn


class NNModel(ABC):
    def __init__(self,
                 lr=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 epoches=100,
                 log_wise=10):

        self.epoches = epoches
        self.log_wise = log_wise
        self.optr = None

    @abstractmethod
    def predict_proba(self, x: jnp.ndarray, params):
        pass

    def fit(self, x_train, y_train_proba, x_test, y_test_proba):

        @jit
        def _acc(y_true_proba, y_pred_proba):
            y_true = jnp.argmax(y_true_proba, axis=1)
            y_pred = jnp.argmax(y_pred_proba, axis=1)
            return jnp.mean(y_true == y_pred)

        _loss = lambda params: cross_entropy_loss(y_train_proba, self.predict_proba(x_train, params))
        _loss = jit(_loss)  # accelerate loss function by JIT

        self.optr.open(_loss)

        _tloss = lambda params: cross_entropy_loss(y_test_proba, self.predict_proba(x_test, params))
        _tloss = jit(_tloss)

        acc, loss, tacc, tloss = [], [], [], []  # train acc, train loss, test acc, test loss

        for _ in range(self.epoches):
            loss.append(_loss(self.optr.get_params()))
            tloss.append(_tloss(self.optr.get_params()))

            self.optr.update()

            acc.append(_acc(y_train_proba, self.predict_proba(x_train, self.optr.get_params())))
            tacc.append(_acc(y_test_proba, self.predict_proba(x_test, self.optr.get_params())))
            if self.optr.get_steps() % self.log_wise == 0:
                print(f'>> epoch: {self.optr.get_steps()}, train acc: {acc[-1]}, train loss: {loss[-1]}; test acc: {tacc[-1]}, test loss: {tloss[-1]}')

        return acc, loss, tacc, tloss


__all__ = ['Conv', 'Rnn', 'NNModel']

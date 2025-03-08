from abc import ABC, abstractmethod
from jax import jit

from .utils import cross_entropy_loss


class Loss(ABC):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        # self.short_batch = short_batch  # use 'drop' here

    @abstractmethod
    def loss(self, params, x, y_true):
        pass

    def load(self, x, y_true):
        self.cnt = 0
        self.losses = []

        for ix in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):  # drop last batch, used when: dataset size >> batch size
            bx = x[ix:ix + self.batch_size]
            by = y_true[ix:ix + self.batch_size]

            self.losses.append(lambda params: jit(self.loss(params, bx, by)))  # append loss function for each batch & use jit compile

    def next(self):
        self.cnt = (self.cnt + 1) % len(self.losses)  # cycle iter
        return self.losses[self.cnt]


class CrossEntropyLoss(Loss):
    def __init__(self, predict_proba, batch_size=32):
        super(CrossEntropyLoss, self).__init__(batch_size)

        self.predict_proba = predict_proba

    def loss(self, params, x, y_true):
        return cross_entropy_loss(y_true, self.predict_proba(x, params))

import jax.numpy as jnp
from jax import random

from plugins.minitorch.nn import Rnn, Dense
from plugins.minitorch import Initer
from plugins.minitorch.optimizer import Adam
from plugins.minitorch.loss import CrossEntropyLoss
from plugins.minitorch.utils import softmax

from plugins.lrkit.clfs import Clfs, timing


class lstm(Clfs):
    def __init__(self, lr, epoches, batch_size, depth=1):
        super(lstm, self).__init__()

        self.config = {
            'lstm:0': Rnn.get_lstm(128, 9, 64),
            'fc:0': Dense.get_linear(64, 6),
        }

        self.epoches = epoches
        self.lr = lr
        self.batch_size = batch_size
        self.losr = CrossEntropyLoss(self.forward)

    def forward(self, x, params, train=False):
        res = jnp.transpose(x, (2, 0, 1))
        res, _, _ = Rnn.lstm(res, params['lstm:0'], self.config['lstm:0'])
        res = res[-1]

        res = Dense.linear(res, params['fc:0'])

        return softmax(res)

    @timing
    def predict_proba(self, x):
        return self.forward(params=self.optr.get_params(), x=x, train=False)

    @timing
    def fit(self, x, y):
        self.optr = Adam(Initer(self.config, random.PRNGKey(42))(), lr=self.lr, batch_size=self.batch_size)

        _loss = self.losr.get_loss(True)
        self.optr.open(_loss, x, y)

        _tloss = self.losr.get_loss(False)

        log_wise = self.epoches // 10 if self.epoches >= 10 else self.epoches
        for cnt in range(self.epoches):
            if (cnt + 1) % log_wise == 0:
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}')

            self.optr.update()

        self.optr.close()


class gru(Clfs):
    def __init__(self, lr, epoches, batch_size, depth=1):
        super(gru, self).__init__()

        self.config = {
            'gru:0': Rnn.get_gru(128, 9, 64),
            'fc:0': Dense.get_linear(64, 6),
        }

        self.epoches = epoches
        self.lr = lr
        self.batch_size = batch_size
        self.losr = CrossEntropyLoss(self.forward)

    def forward(self, x, params, train=False):
        res = jnp.transpose(x, (2, 0, 1))
        res, _ = Rnn.gru(res, params['gru:0'], self.config['gru:0'])
        res = res[-1]

        res = Dense.linear(res, params['fc:0'])

        return softmax(res)

    @timing
    def predict_proba(self, x):
        return self.forward(params=self.optr.get_params(), x=x, train=False)

    @timing
    def fit(self, x, y):
        self.optr = Adam(Initer(self.config, random.PRNGKey(42))(), lr=self.lr, batch_size=self.batch_size)
        _loss = self.losr.get_loss(True)
        self.optr.open(_loss, x, y)

        _tloss = self.losr.get_loss(False)

        log_wise = self.epoches // 10 if self.epoches >= 10 else self.epoches
        for cnt in range(self.epoches):
            if (cnt + 1) % log_wise == 0:
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}')

            self.optr.update()

        self.optr.close()


class conv3x3(Clfs):
    pass

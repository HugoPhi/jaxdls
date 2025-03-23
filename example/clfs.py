import jax.numpy as jnp
from jax import random
import jax

from plugins.minitorch.nn import Conv, Rnn, Dense
from plugins.minitorch import Initer
from plugins.minitorch.optimizer import Adam
from plugins.minitorch.loss import CrossEntropyLoss
from plugins.minitorch.utils import softmax

from plugins.lrkit import ClfTrait, timing


def acc(y_true, y_pred):
    return jnp.mean(jnp.argmax(y_true, axis=1) == jnp.argmax(y_pred, axis=1))


'''
rnn based model
'''


class lstm(ClfTrait):
    def __init__(self, lr, epoches, batch_size):
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
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}, acc: {acc(y, self.predict_proba(x))}')

            self.optr.update()

        self.optr.close()
        jax.clear_caches()


class gru(ClfTrait):
    def __init__(self, lr, epoches, batch_size):
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
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}, acc: {acc(y, self.predict_proba(x))}')

            self.optr.update()

        self.optr.close()
        jax.clear_caches()


class bilstm(ClfTrait):
    def __init__(self, lr, epoches, batch_size):
        super(bilstm, self).__init__()

        self.config = {
            'bilstm:0': Rnn.get_lstm(128, 9, 64),
            'fc:0': Dense.get_linear(128, 6),
        }

        self.epoches = epoches
        self.lr = lr
        self.batch_size = batch_size
        self.losr = CrossEntropyLoss(self.forward)

    def forward(self, x, params, train=False):
        res = jnp.transpose(x, (2, 0, 1))
        res, _, _ = Rnn.bilstm(res, params['bilstm:0'], self.config['bilstm:0'])
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
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}, acc: {acc(y, self.predict_proba(x))}')

            self.optr.update()

        self.optr.close()
        jax.clear_caches()


class bigru(ClfTrait):
    def __init__(self, lr, epoches, batch_size):
        super(bigru, self).__init__()

        self.config = {
            'bigru:0': Rnn.get_bigru(128, 9, 64),
            'fc:0': Dense.get_linear(128, 6),
        }

        self.epoches = epoches
        self.lr = lr
        self.batch_size = batch_size
        self.losr = CrossEntropyLoss(self.forward)

    def forward(self, x, params, train=False):
        res = jnp.transpose(x, (2, 0, 1))
        res, _ = Rnn.bigru(res, params['bigru:0'], self.config['bigru:0'])
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
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}, acc: {acc(y, self.predict_proba(x))}')

            self.optr.update()

        self.optr.close()
        jax.clear_caches()


'''
cnn based model
'''


class conv1dx3(ClfTrait):
    def __init__(self, lr, epoches, batch_size):
        super(conv1dx3, self).__init__()

        self.config = {
            'conv1d:00': Conv.get_conv1d(9, 16, (3,)),   # 128 -> 126
            'conv1d:01': Conv.get_conv1d(16, 16, (3,)),  # 126 -> 124
            'maxpooling1d:0': Conv.get_max_pool1d(2),  # 124 -> 62
            'conv1d:10': Conv.get_conv1d(16, 32, (3,)),  # 62 -> 60
            'conv1d:11': Conv.get_conv1d(32, 32, (3,)),  # 60 -> 58
            'maxpooling1d:1': Conv.get_max_pool1d(2),  # 58 -> 29
            'conv1d:20': Conv.get_conv1d(32, 64, (3,)),  # 29 -> 27
            'conv1d:21': Conv.get_conv1d(64, 64, (3,)),  # 27 -> 25
            'maxpooling1d:2': Conv.get_max_pool1d(2),  # 25 -> 12
            'fc:0': Dense.get_linear(12 * 64, 256),  # 64 x 12 = 768
            'fc:1': Dense.get_linear(256, 64),
            'fc:2': Dense.get_linear(64, 6)
        }

        self.epoches = epoches
        self.lr = lr
        self.batch_size = batch_size
        self.losr = CrossEntropyLoss(self.forward)

    def conv_block(self, x, params, id):
        res = Conv.conv1d(x, params[f'conv1d:{id}0'], self.config[f'conv1d:{id}0'])
        res = Conv.conv1d(res, params[f'conv1d:{id}1'], self.config[f'conv1d:{id}1'])
        res = Conv.max_pooling1d(res, self.config[f'maxpooling1d:{id}'])

        return res

    def forward(self, x, params, train=False):
        res = self.conv_block(x, params, 0)
        res = self.conv_block(res, params, 1)
        res = self.conv_block(res, params, 2)

        res = res.reshape(res.shape[0], -1)

        res = Dense.linear(res, params['fc:0'])
        res = Dense.linear(res, params['fc:1'])
        res = Dense.linear(res, params['fc:2'])

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
                print(f'====> Epoch {cnt + 1}/{self.epoches}, loss: {_tloss(self.optr.get_params(), x, y)}, acc: {acc(y, self.predict_proba(x))}')

            self.optr.update()

        self.optr.close()
        jax.clear_caches()

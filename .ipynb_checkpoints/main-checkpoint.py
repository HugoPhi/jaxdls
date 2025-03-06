import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

from plugins.minitorch.nn import Rnn, NNModel
from plugins.minitorch.optimizer import Adam
from plugins.minitorch.initer import Initer
from plugins.minitorch.utils import softmax

from data_process import X_train, X_test, y_train, y_test

key = random.PRNGKey(0)


class LSTM(NNModel):
    def __init__(self, lr, epoches):
        super().__init__(lr=lr, epoches=epoches)

        self.config = {
            'lstm:0': {
                'time_steps': 128,
                'input_dim': 9,
                'hidden_dim': 64,
            },
            'fc:0': {
                'input_dim': 64,
                'output_dim': 6,
            }
        }

        initer = Initer(self.config, key)
        self.optr = Adam(initer(), lr=lr)

    def predict_proba(self, x, params):
        res, _, _ = Rnn.lstm_cell(x,
                                  jnp.zeros((x.shape[1], self.config['lstm:0']['hidden_dim'])),
                                  jnp.zeros((x.shape[1], self.config['lstm:0']['hidden_dim'])),
                                  params['lstm:0']['Ws'], params['lstm:0']['Us'], params['lstm:0']['Bs'])

        res = res[-1]
        res = res @ params['fc:0']['w'] + params['fc:0']['b']

        return softmax(res)


epochs = 250
model = LSTM(lr=0.015, epoches=epochs)
acc, loss, tacc, tloss = model.fit(
    x_train=X_train,
    y_train_proba=y_train,
    x_test=X_test,
    y_test_proba=y_test,
)


fig, ax1 = plt.subplots()

plt.rcParams['font.family'] = 'Noto Serif SC'
plt.rcParams['font.sans-serif'] = ['Noto Serif SC']

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(range(epochs), acc, color=color, label='Train Accuracy', linestyle='-')
ax1.plot(range(epochs), tacc, color=color, label='Test Accuracy', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
ax2.plot(range(epochs), loss, color=color, label='Train Loss', linestyle='-')
ax2.plot(range(epochs), tloss, color=color, label='Test Loss', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right')

plt.title('Training and Testing Accuracy and Loss over Epochs')
fig.tight_layout()
plt.show()

print(f'final train, test acc : {acc[-1]}, {tacc[-1]}')
print(f'final train, test loss: {loss[-1]}, {tloss[-1]}')

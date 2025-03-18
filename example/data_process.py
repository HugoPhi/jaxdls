# Download latest version
from jax import random
import jax.numpy as jnp
import numpy as np
import pandas as pd
import kagglehub

# path = kagglehub.dataset_download("drsaeedmohsen/ucihar-dataset") + '/UCI-HAR Dataset'
path = '/root/.cache/kagglehub/datasets/drsaeedmohsen/ucihar-dataset/versions/1' + '/UCI-HAR Dataset'
print("Path to dataset files:", path)


key = random.PRNGKey(273)

TRAIN_PATH = path + '/train/Inertial Signals/'
TEST_PATH = path + '/test/Inertial Signals/'
PREFIXS = [
    'body_acc_x_',
    'body_acc_y_',
    'body_acc_z_',
    'body_gyro_x_',
    'body_gyro_y_',
    'body_gyro_z_',
    'total_acc_x_',
    'total_acc_y_',
    'total_acc_z_',
]

X_train = []
for prefix in PREFIXS:
    X_train.append(pd.read_csv(TRAIN_PATH + prefix + 'train.txt', header=None, sep=r'\s+').to_numpy())

X_train = np.transpose(np.array(X_train), (1, 0, 2))
X_train = jnp.array(X_train)

X_test = []
for prefix in PREFIXS:
    X_test.append(pd.read_csv(TEST_PATH + prefix + 'test.txt', header=None, sep=r'\s+').to_numpy())
X_test = np.transpose(np.array(X_test), (1, 0, 2))
X_test = jnp.array(X_test)


y_train = jnp.array(pd.read_csv(path + '/train/y_train.txt', header=None).to_numpy().squeeze() - 1)
y_test = jnp.array(pd.read_csv(path + '/test/y_test.txt', header=None).to_numpy().squeeze() - 1)

# 将标签转换为 one-hot 编码


def one_hot(y: jnp.ndarray, num_class: int):
    res = jnp.zeros((y.shape[0], num_class))
    res = res.at[jnp.arange(y.shape[0]), y].set(1)
    return res


y_train = one_hot(y_train, 6)
y_test = one_hot(y_test, 6)

# Suffle
TRAIN = None
TEST = None

Shuffle = True

if Shuffle:
    shuffle_kernel = random.permutation(key, (X_train.shape[0]))
    X_train = X_train[shuffle_kernel][:TRAIN]
    y_train = y_train[shuffle_kernel][:TRAIN]
    shuffle_kernel = random.permutation(key, (X_test.shape[0]))
    X_test = X_test[shuffle_kernel][:TEST]
    y_test = y_test[shuffle_kernel][:TEST]
else:
    X_train = X_train[:TRAIN]
    y_train = y_train[:TRAIN]
    X_test = X_test[:TEST]
    y_test = y_test[:TEST]

# X_train = jnp.transpose(X_train, (2, 0, 1))
# X_test = jnp.transpose(X_test, (2, 0, 1))

print('X_train 形状:', X_train.shape)  # 应为 (7352, 9, 128)
print('y_train 形状:', y_train.shape)  # 应为 (7352, 6)
print('X_test  形状:', X_test.shape)
print('y_test  形状:', y_test.shape)

# 🔥 Classic DeepLearning Models by Jax

<p align="center">
  <img src="./assets/mlp.svg" alt="MLP on MNIST" width="45%">
  <img src="./assets/LeNet.svg" alt="LeNet on MNIST" width="45%">
</p>

<p align="center">
On MNIST: (a) acc[96.80%] & loss vs. epochs for mlp; (b) acc[97.86%] & loss vs. epochs for LeNet[1]
</p>


## # models implemented in this project

- Linear Regression
  - Self Made Gauss-Noise of a Function.
- Logistic Regression
  - Iris.
- KNN
  - CIFAR-10.
- MLP
  - MNIST.
  - CIFAR-10.
- LeNet
  - MNIST.
  - CIFAR-10.
- LSTM
  - UCI HAR.
- GRU[3]
  - UCI HAR.
- Transformer[4]
  - WMT15(TODO).
- Nerual ODE[5]
  - MNIST(TODO).
- VAE
  - MNIST(TODO).

## # unit tests

- How to Use Jax Gradient.
- Kaiming Initialization[2] used in MLP & Conv (with derivation).
- Conv Operation & Time Performances.
- RNN Cells(Basic, LSTM, GRU) & Time Performances.
- Runge-Kuta solver for Neural ODE.

## # Plugin: Mini-torch

- nn
  - Conv
    - Conv2d
    - MaxPooling2d
  - RnnCell
    - Basic rnn kernel
    - LSTM kernel
    - GRU kernel
- Optimizer
  - Batch GD
  - Mini-Batch GD
  - Adam[6]
- utils
  - softmax
  - cross_entropy_loss
  - mean_square_error
  - l1_regularization
  - l2_regularization

# Reference

[1] [Y. LeCun et al., Backpropagation Applied to Handwritten Zip Code Recognition (1989)](https://ieeexplore.ieee.org/document/6795724)  
[2] [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852)  
[3] [Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. Proceedings of the 30th International Conference on Machine Learning (ICML), 1310–1318.](https://arxiv.org/abs/1211.5063)    
[4] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS 2017).](https://arxiv.org/abs/1706.03762)   
[5] [Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems, 31.](https://arxiv.org/abs/1806.07366?spm=5176.28103460.0.0.40f7451eXLzPoY&file=1806.07366)   
[6] [Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. Proceedings of the International Conference on Learning Representations (ICLR).](https://arxiv.org/abs/1412.6980?spm=5176.28103460.0.0.40f7451eXLzPoY&file=1412.6980)   
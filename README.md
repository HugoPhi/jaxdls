# 🔥 Classic DeepLearning Models by Jax

> <ins>💡: The Latest Framework test case is under: `./example/`.</ins>  

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
  - WMT15. <mark>TODO</mark>
- Nerual ODE[5]
  - MNIST. <mark>TODO</mark>
- VAE
  - MNIST. <mark>TODO</mark>

## # NoteBook Docs

Some small tests for debug during the development of this project:   

- How to Use Jax Gradient, <ins>*Ideas about how I manage parameters in this Framework*</ins>. <mark>TODO</mark>
- When to use JIT in Jax? <ins>*About Time & Space*</ins>  <mark>TODO</mark>
- Kaiming Initialization[2] used in MLP & Conv, <ins>*With math derivation*</ins>  
- Difference between Conv2d Operation by python loop and by **jax.lax**.
- Dropout mechanic impl, <ins>*About Seed in Jax*.</ins>
- Runge-Kuta solver for Neural ODE.

## # Plugin: Mini-torch

- nn
  - Model (Base Class for Nerual Networks, like nn.Module in torch)
  - Conv
    - Conv1d, Conv2d, Conv3d
    - MaxPooling1d, MaxPooling2d, MaxPooling3d
    - BatchNorm <mark>TODO</mark>
  - RnnCell
    - Basic rnn kernel
    - LSTM kernel
    - GRU kernel
  - FC
    - Dropout
    - Linear
- Optimizer
  - Raw GD
  - Momentum
  - Nesterov(NAG)
  - AdaGrad
  - RMSProp
  - AdaDelta
  - Adam[6]
- Utils
  - sigmoid
  - one hot
  - softmax
  - cross_entropy_loss
  - mean_square_error
  - l1_regularization
  - l2_regularization

# Reference

[[1](https://ieeexplore.ieee.org/document/6795724)] Y. LeCun et al., Backpropagation Applied to Handwritten Zip Code Recognition (1989).    
[[2](https://arxiv.org/abs/1502.01852)] Delving Deep into Rectifiers (He et al., 2015)   
[[3](https://arxiv.org/abs/1211.5063)] Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. Proceedings of the 30th International Conference on Machine Learning (ICML), 1310–1318.    
[[4](https://arxiv.org/abs/1706.03762)] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS 2017).   
[[5](https://arxiv.org/abs/1806.07366?spm=5176.28103460.0.0.40f7451eXLzPoY&file=1806.07366)] <ins>Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems, 31.</ins>   
[[6](https://arxiv.org/abs/1412.6980?spm=5176.28103460.0.0.40f7451eXLzPoY&file=1412.6980)] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. Proceedings of the International Conference on Learning Representations (ICLR).   

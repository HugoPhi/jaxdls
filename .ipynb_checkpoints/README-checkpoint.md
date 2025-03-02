# 🔥 Classic DeepLearning Models by Jax

<center>
<img src="./assets/mlp.png" alt="First Image" width="45%">
<img src="./assets/LeNet.png" alt="Second Image" width="45%">
</center>

<center>
On MNIST: (a) acc[96.80%] & loss vs. epochs for mlp; (b) acc[97.10%] & loss vs. epochs for LaNet[1]
</center>

## # models implemented in this project

- Linear Regression
    - Self Made Gauss-Noise of a Function.
- Logistic Regression
    - Iris.
- KNN
    - CIFAR-10.
- MLP.
    - MNIST.
- LeNet
    - MNIST.
- LSTM(TODO)
    - UCI HAR.

## # unit tests

- How to Use Jax Gradient.
- Kaiming Initialization[2] used in MLP & Conv (with derivation).
- Conv Operation & Time Performances.
- RNN Cells(Basic, LSTM, GRU[3]) & Time Performances.

# Reference

[1] [Y. LeCun et al., Backpropagation Applied to Handwritten Zip Code Recognition (1989)](https://ieeexplore.ieee.org/document/6795724)  
[2] [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852)  
[3] [Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. Proceedings of the 30th International Conference on Machine Learning (ICML), 1310–1318.](https://arxiv.org/abs/1211.5063)
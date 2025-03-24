# 🔥 Classic DeepLearning Models by Jax

> <ins>💡: The Latest Framework test case is under: [`./example/`](./example/README.md).</ins>  

<p align="center">
  <img src="./assets/mlp.svg" alt="MLP on MNIST" width="45%">
  <img src="./assets/LeNet.svg" alt="LeNet on MNIST" width="45%">
</p>

<p align="center">
On MNIST: (a) acc[96.80%] & loss vs. epochs for mlp; (b) acc[98.24%] & loss vs. epochs for LeNet  
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
- LeNet[[1](#reference)]
  - MNIST.
  - CIFAR-10.
- LSTM
  - UCI HAR.
- GRU[[2](#reference)]
  - UCI HAR.
- Transformer[[3](#reference)]
  - WMT15. <mark>TODO</mark>
- Nerual ODE[[4](#reference)]
  - MNIST. <mark>TODO</mark>
- VAE[[5](#reference)]
  - MNIST. 

## # Plugins

- [Minitorch](), branch: `main`.
- [LrKit](https://github.com/HugoPhi/lrkit.git), branch: `jax`.

#### @ Number of Codes

Last update: 2025.03.14.   

```text
     236 text files.
     135 unique files.                              
     138 files ignored.

github.com/AlDanial/cloc v 1.98  T=0.05 s (2810.5 files/s, 307803.1 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          33           1689           3297           3177
Jupyter Notebook                21              0           3947           1913
Text                             6              1              0            301
CSV                             68              0              0            203
Markdown                         5             40              0            198
TOML                             2              3              0             16
-------------------------------------------------------------------------------
SUM:                           135           1733           7244           5808
-------------------------------------------------------------------------------
```

# Reference

[[1](https://ieeexplore.ieee.org/document/6795724)] LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard, R., Hubbard, W., & Jackel, L. (1989). Backpropagation Applied to Handwritten Zip Code Recognition. Neural Computation, 1(4), 541–551.   
[[2](https://arxiv.org/abs/1211.5063)] Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. In Proceedings of the 30th International Conference on Machine Learning (ICML) (pp. 1310–1318).   
[[3](https://arxiv.org/abs/1706.03762)] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, ., & Polosukhin, I. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS).   
[[4](https://arxiv.org/abs/1806.07366?spm=5176.28103460.0.0.40f7451eXLzPoY&file=1806.07366)] <ins>Chen, T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. In Advances in Neural Information Processing Systems (NeurIPS).</ins>   
[[5](https://arxiv.org/abs/1312.6114)] Kingma, D., & Welling, M. (2014). Auto-Encoding Variational Bayes. In International Conference on Learning Representations (ICLR). 

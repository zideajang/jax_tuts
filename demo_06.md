JAX 最初由谷歌大脑团队的 Matt Johnson、Roy Frostig、Dougal Maclaurin 和 Chris Leary 等人发起


JAX 面向人群是研究院，而不是开发人员，这一点想大家在开始了解这个库需要清楚的一点。

#### JAX 是什么
JAX 的前身是 Autograd ，也就是说 JAX 是 Autograd 升级版本，JAX 可以对 Python 和 NumPy 程序进行自动微分。可以通过 Python的大量特征子集进行区分，包括循环、分支、递归和闭包语句进行自动求导，也可以求三阶导数(三阶导数是由原函数导数的导数的导数。 所谓三阶导数，即原函数导数的导数的导数，将原函数进行三次求导)。通过 grad ，JAX 支持反向模式和正向模式的求导，而且这两种模式可以任意组合成任何顺序，具有一定灵活性。

#### 为什么说 JAX 是 numpy 终结者
numpy 是一个科学计算库，就是在今天，就是用 tensorflow 和 pytorch 这样炙手可热的深度学习库去实现模型也好、网络也好，也少不了几行用到 numpy 的代码。可见 numpy 的重要性，但是 numpy 诞生时候并没有大量使用 GPU 来支持运算，所以 numpy 天生

#### JAX 优点
我们大家都已经用惯了 python，而且 python 对于初学者来说容易上手。


#### JAX 安装
```
pip install --upgrade jax jaxlib
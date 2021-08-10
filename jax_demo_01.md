

##   JAX 的出现让自己实现一个深度学习的框架不再那么绕远了

### 有关 JAX 的概述
####  JAX 是什么
JAX 的前身是 Autograd ，也就是说 JAX 是 Autograd 升级版本，JAX 可以对 Python 和 NumPy 程序进行自动微分。可以通过 Python的大量特征子集进行区分，包括循环、分支、递归和闭包语句进行自动求导，也可以求三阶导数(三阶导数是由原函数导数的导数的导数。 所谓三阶导数，即原函数导数的导数的导数，将原函数进行三次求导)。通过 grad ，JAX 支持反向模式和正向模式的求导，而且这两种模式可以任意组合成任何顺序，具有一定灵活性。

#### JAX 面向的人群
JAX 相对于 Tensorflow 和 Pytorch 还是显得比较原始(底层)，许多东西还需自己去实现，可能你会问有必要自己去实现深度学习框架吗? 自己去实现好处就是出现问题更好控制，对于不同任务定制化更强，所以 JAX 是面向研究人员，而不是开发人员，这一点想大家在开始了解这个库需要清楚的一点。

#### 学习 JAX 动机
- 最求性能，当用到既有机器学习框架遇到性能的瓶颈，而又对底层 c++ 和 GPU 结构原理了解不足，可以考虑一下 JAX 来重构自己模型(自己还没有去尝试)
- 自己想要实现一个基于 python 的深度学习框架，可以考虑一下 JAX

要完成一个大规模数据

- 硬件加速
- 自动求导来进行优化运算
- 融合操作 ，例如 np.sum((preds - targets) ** 2)
- 并行处理数据和计算

#### JAX 安装
```
pip install --upgrade jax jaxlib
```

安装 GPU

```shell
pip install --upgrade jax jaxlib==0.1.61+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```



### JAX 可以看成运行在 GPU 和 TPU 上 Numpy 

Numpy 是一个科学计算库，就是在今天，就是用 tensorflow 和 pytorch 这样炙手可热的深度学习库去实现模型也好、网络也好，也少不了几行用到 numpy 的代码。可见 numpy 的重要性，但是 numpy 诞生时候并没有大量使用 GPU 来支持运算，所以 numpy 的程序是无法跑在 GPU，但是 JAX 其实并不是 numpy，只是借鉴 numpy API，让开发人员用起来感觉在用 Numpy，无差别地使用 numpy。



```python
import numpy as np
```
引入 numpy ，为了方便使用为引入 numpy 起一个别名 np。

```python
import numpy as np
x = np.random.rand(2000,2000)
print(x)
```

```

    [[0.56745022 0.4247945  0.32374621 ... 0.72424614 0.31471484 0.75709393]
     [0.76504917 0.41393967 0.1195595  ... 0.27311255 0.36763284 0.39811399]
     [0.30034904 0.8224698  0.0160814  ... 0.75720634 0.72237672 0.09741124]
     ...
     [0.14822982 0.918704   0.22328525 ... 0.67143212 0.91682163 0.65214596]
     [0.25847224 0.7675988  0.64836721 ... 0.19096599 0.89869396 0.22051008]
     [0.23031364 0.60925244 0.72548038 ... 0.63396252 0.13415147 0.0674989 ]]
```


```python
2 * x
```

这样直观进行矩阵运算，例如给 x 每个元素都乘以 2 可以用上面这样直观操作，而无需遍历矩阵每个元素

```
    array([[1.13490044, 0.849589  , 0.64749241, ..., 1.44849228, 0.62942968,
            1.51418785],
           [1.53009834, 0.82787934, 0.239119  , ..., 0.54622511, 0.73526569,
            0.79622798],
           [0.60069808, 1.6449396 , 0.03216279, ..., 1.51441268, 1.44475343,
            0.19482249],
           ...,
           [0.29645964, 1.83740799, 0.4465705 , ..., 1.34286423, 1.83364326,
            1.30429192],
           [0.51694448, 1.5351976 , 1.29673443, ..., 0.38193199, 1.79738792,
            0.44102015],
           [0.46062729, 1.21850487, 1.45096075, ..., 1.26792504, 0.26830294,
            0.1349978 ]])

```


```python
np.sin(x)
```

对于一些复杂的运算例如 `np.sin` numpy 也应付自如。

```
    array([[0.53748363, 0.41213356, 0.31812038, ..., 0.66257099, 0.30954533,
            0.68681211],
           [0.69257247, 0.40221938, 0.11927486, ..., 0.26972993, 0.35940746,
            0.38768052],
           [0.29585364, 0.73282855, 0.0160807 , ..., 0.68689382, 0.66116964,
            0.09725726],
           ...,
           [0.14768759, 0.79481581, 0.2214345 , ..., 0.62210787, 0.79367208,
            0.60689338],
           [0.25560384, 0.69440939, 0.60388576, ..., 0.18980742, 0.78251439,
            0.21872738],
           [0.2282829 , 0.57225456, 0.66349493, ..., 0.59234195, 0.13374945,
            0.06744766]])
```



```python
x - x.mean(0)
```



```
    array([[ 0.05966959, -0.07397188, -0.18537367, ...,  0.21733322,
            -0.18467283,  0.25997255],
           [ 0.25726854, -0.08482671, -0.38956037, ..., -0.23380037,
            -0.13175483, -0.09900739],
           [-0.20743159,  0.32370341, -0.49303848, ...,  0.25029342,
             0.22298905, -0.39971013],
           ...,
           [-0.35955081,  0.41993761, -0.28583463, ...,  0.1645192 ,
             0.41743396,  0.15502459],
           [-0.24930839,  0.26883241,  0.13924734, ..., -0.31594693,
             0.39930629, -0.2766113 ],
           [-0.27746699,  0.11048605,  0.2163605 , ...,  0.1270496 ,
            -0.3652362 , -0.42962248]])

```


```python
np.dot(x,x)
```

矩阵间点乘也十分方便

```
    array([[499.08919102, 490.98247709, 495.18751355, ..., 498.40635521,
            494.50937914, 485.34695773],
           [510.29685902, 499.95239357, 511.85978277, ..., 509.82817989,
            495.05226925, 507.41925595],
           [502.82328413, 501.8213885 , 506.67580735, ..., 508.35889233,
            492.64972834, 493.06081799],
           ...,
           [502.20453325, 496.38140482, 508.98725444, ..., 505.05666502,
            490.64576912, 491.95629717],
           [515.66634283, 498.26014692, 516.70676734, ..., 508.06152946,
            506.435225  , 500.36645682],
           [509.67692906, 502.64662385, 509.47906271, ..., 509.0583251 ,
            505.48856182, 493.5220343 ]])
```



接下来我们看一看 jax 的 numpy 模块提供方法类似于 numpy 的方法，我们对比去上面 numpy 操作都用 `jax.numpy` 去实现一遍。

```python
import jax.numpy as jnp
```


```python
y = jnp.array(x)
```


```python
y
```

将 numpy 对象来 `DeviceArray`

```
    DeviceArray([[0.5674502 , 0.4247945 , 0.3237462 , ..., 0.72424614,
                  0.31471485, 0.7570939 ],
                 [0.76504916, 0.41393968, 0.1195595 , ..., 0.27311257,
                  0.36763284, 0.398114  ],
                 [0.30034903, 0.8224698 , 0.0160814 , ..., 0.7572063 ,
                  0.7223767 , 0.09741125],
                 ...,
                 [0.14822982, 0.918704  , 0.22328524, ..., 0.67143214,
                  0.9168216 , 0.652146  ],
                 [0.25847223, 0.7675988 , 0.6483672 , ..., 0.190966  ,
                  0.898694  , 0.22051008],
                 [0.23031364, 0.60925245, 0.7254804 , ..., 0.6339625 ,
                  0.13415147, 0.0674989 ]], dtype=float32)
```



```python
2 * y
```


```

    DeviceArray([[1.1349005 , 0.849589  , 0.6474924 , ..., 1.4484923 ,
                  0.6294297 , 1.5141878 ],
                 [1.5300983 , 0.82787937, 0.23911901, ..., 0.54622513,
                  0.7352657 , 0.796228  ],
                 [0.60069805, 1.6449395 , 0.03216279, ..., 1.5144126 ,
                  1.4447534 , 0.19482249],
                 ...,
                 [0.29645965, 1.837408  , 0.4465705 , ..., 1.3428643 ,
                  1.8336432 , 1.304292  ],
                 [0.51694447, 1.5351976 , 1.2967345 , ..., 0.381932  ,
                  1.797388  , 0.44102016],
                 [0.4606273 , 1.2185049 , 1.4509608 , ..., 1.267925  ,
                  0.26830295, 0.1349978 ]], dtype=float32)
```



```python
jnp.sin(y)
```


```

    DeviceArray([[0.53748363, 0.41213354, 0.31812036, ..., 0.662571  ,
                  0.30954534, 0.6868121 ],
                 [0.6925725 , 0.40221938, 0.11927487, ..., 0.26972994,
                  0.35940745, 0.38768053],
                 [0.2958536 , 0.73282856, 0.0160807 , ..., 0.6868938 ,
                  0.66116965, 0.09725726],
                 ...,
                 [0.1476876 , 0.79481584, 0.2214345 , ..., 0.6221079 ,
                  0.7936721 , 0.6068934 ],
                 [0.25560382, 0.69440943, 0.60388577, ..., 0.18980742,
                  0.78251445, 0.21872738],
                 [0.2282829 , 0.5722546 , 0.66349494, ..., 0.59234196,
                  0.13374946, 0.06744765]], dtype=float32)
```



```python
y - y.mean(0)
```


```

    DeviceArray([[ 0.05966955, -0.0739719 , -0.18537366, ...,  0.2173332 ,
                  -0.18467283,  0.2599725 ],
                 [ 0.2572685 , -0.08482671, -0.38956037, ..., -0.23380038,
                  -0.13175485, -0.0990074 ],
                 [-0.20743164,  0.32370338, -0.49303848, ...,  0.25029337,
                   0.22298902, -0.39971015],
                 ...,
                 [-0.35955083,  0.41993758, -0.2858346 , ...,  0.16451919,
                   0.41743392,  0.15502459],
                 [-0.24930844,  0.26883242,  0.13924736, ..., -0.31594694,
                   0.3993063 , -0.27661133],
                 [-0.277467  ,  0.11048606,  0.21636051, ...,  0.12704957,
                  -0.36523622, -0.4296225 ]], dtype=float32)
```



```python
jnp.dot(y,y)
```


```

    DeviceArray([[499.08923, 490.98248, 495.18756, ..., 498.4064 , 494.50937,
                  485.347  ],
                 [510.2968 , 499.95236, 511.85983, ..., 509.8281 , 495.0523 ,
                  507.4191 ],
                 [502.82324, 501.82147, 506.67572, ..., 508.35886, 492.64972,
                  493.06076],
                 ...,
                 [502.20465, 496.3814 , 508.9873 , ..., 505.0567 , 490.6458 ,
                  491.95618],
                 [515.66626, 498.2601 , 516.70667, ..., 508.06168, 506.43524,
                  500.3665 ],
                 [509.67685, 502.64664, 509.47913, ..., 509.0583 , 505.48856,
                  493.52206]], dtype=float32)

```


```python
%timeit np.dot(x,x)
%timeit jnp.dot(y,y)
```
```
    47.2 ms ± 5.78 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    2.16 ms ± 21.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
从执行两个方法耗时对比来看，`jpn.dot` 还是具有很大优势。

### JIT compilation(即使编译)
为了利用 XLA  的强大功能，将代码编译到 XLA 内核中。这就是 jit 发挥作用的地方。要使用 XLA 和 jit，可以使用 jit() 函数或 @jit 注释。

```python
def f(x):
    for i in range(10):
        x -= 0.1 * x
    return x
```

这里我们定义函数`f(x)`，函数本身并没有实际意义，旨在说明 JIT compilation

```python
f(x)
```
我们可以用 numpy 来运行执行对矩阵运算

```

    array([[0.19785766, 0.14811668, 0.11288332, ..., 0.25252901, 0.10973428,
            0.26398233],
           [0.26675615, 0.14433184, 0.04168782, ..., 0.09522846, 0.12818565,
            0.13881376],
           [0.10472524, 0.28677749, 0.00560724, ..., 0.26402153, 0.25187719,
            0.0339652 ],
           ...,
           [0.05168454, 0.32033228, 0.07785475, ..., 0.2341139 , 0.31967594,
            0.22738924],
           [0.0901237 , 0.26764515, 0.22607167, ..., 0.06658573, 0.31335521,
            0.07688711],
           [0.0803054 , 0.21243319, 0.25295937, ..., 0.22104906, 0.04677572,
            0.02353541]])

```


```python
f(y)
```
可以用 jax.numpy 来执行这一些对矩阵的操作。


```
    DeviceArray([[0.19785768, 0.1481167 , 0.11288333, ..., 0.25252903,
                  0.10973427, 0.26398236],
                 [0.26675615, 0.14433186, 0.04168782, ..., 0.09522847,
                  0.12818564, 0.13881375],
                 [0.10472523, 0.2867775 , 0.00560724, ..., 0.26402152,
                  0.2518772 , 0.0339652 ],
                 ...,
                 [0.05168454, 0.32033232, 0.07785475, ..., 0.23411393,
                  0.31967595, 0.22738926],
                 [0.09012369, 0.26764515, 0.22607167, ..., 0.06658573,
                  0.31335524, 0.07688711],
                 [0.08030539, 0.21243319, 0.25295934, ..., 0.22104907,
                  0.04677573, 0.02353541]], dtype=float32)

```

这是我们来计算一些 `f(y)` 执行耗时，因为是同步执行，所以事件上看上去比较长，接下来我们来用 JIT 来执行这个函数

```python
%timeit f(y)
```
```
    3.42 ms ± 31.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

在使用之前 JIT 之前，我们需要引入 jit 包，使用起来也比较方便，用 `jit` 对函数 `f` 进行包裹一下就得到 JIT 

```python
from jax import jit
g = jit(f)
```


```python
%timeit g(y)
```
```
    88.2 µs ± 560 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

包含多个 numpy 运算的函数可以通过` jax.jit() `进行` just-in-time`编译，变成一个单一的 CUDA 程序后来执行，进一步加快运算速度。


### 自动微分
通过 grad() 函数自动微分，这对深度学习应用非常有用，这样就可以很容易地运行反向传播。在深度学习我们通过梯度去更新参数，所以自动求导是深度学习框架实现的重点也是难点。

```python
def f(x):
    return x * jnp.sin(x)
```
这里定义函数 
$$
f(x) = x \sin(x)
$$

```python
f(3)
```

```
    DeviceArray(0.42336, dtype=float32)
```

对于这个函数求导，我们链式法则和常用函数求导可以得到如下
$$
f^{\prime}(x) = \sin(x) + x \cos(x)
$$
```python
def grad_f(x):
    return jnp.sin(x) + x * jnp.cos(x)
```


```python
grad_f(3)
```


```
    DeviceArray(-2.8288574, dtype=float32)
```



引入 jax 的 grad ，然后 grad 包裹 f 返回一个求导函数，自动求导帮助支持链式求导，将反向传播对于程序设计变得简答，其实深度学习框架难点就在于反向求导

```python
from jax import grad
```


```python
grad_f_jax = grad(f)
grad_f_jax(3.0)
```



```
    DeviceArray(-2.8288574, dtype=float32)
```



### 向量化(vectorization)

vmap 是一种函数转换，JAX 通过 vmap 变换提供了自动向量化算法，大大简化了这种类型的计算，这使得研究人员在处理新算法时无需再去处理批量化的问题。示例如下：




```python
def square(x):
    return jnp.sum(x ** 2)
```

定义 `square` 函数对向量每个元素求平方，然后对这个向量进行求和，可以想一下这是先对向量做 map 然后在做 reduce 的操作。

```python
square(jnp.arange(100))
```


```
    DeviceArray(328350, dtype=int32)
```

为了解释一下 vmap 我们可看如何 numpy 来实现一下什么是 vmap。JAX 的 API 中还有一个转换，可能你还没有意识 vmap() 向量映射的好处。可能熟悉 map 函数式沿着数组轴来操作数组中每一个元素，在 vmap 中不是把循环放在外面，而是把循环推到函数的原始操作中进行，从而获得更好的性能。

```python
x = jnp.arange(100).reshape(10,10)
[square(row) for row in x]
```

```
    [DeviceArray(285, dtype=int32),
     DeviceArray(2185, dtype=int32),
     DeviceArray(6085, dtype=int32),
     DeviceArray(11985, dtype=int32),
     DeviceArray(19885, dtype=int32),
     DeviceArray(29785, dtype=int32),
     DeviceArray(41685, dtype=int32),
     DeviceArray(55585, dtype=int32),
     DeviceArray(71485, dtype=int32),
     DeviceArray(89385, dtype=int32)]
```


```python
from jax import vmap
vmap(square)
```


```
    <function __main__.square(x)>
```

```python
vmap(square)(x)
```

```
    DeviceArray([  285,  2185,  6085, 11985, 19885, 29785, 41685, 55585,
                 71485, 89385], dtype=int32)
```

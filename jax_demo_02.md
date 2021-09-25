深度学习框架新贵—JAX(2)—JAX 能否终结 Numpy 的时代

JAX 继承 Numpy 操作数组 friendly 的 API，不仅于此还可以运行在 GPU 上，同时引入自动求导

用于 Javascript 去解析 Javascript 自己动手写一个 Parser(1)

自己动手一步一步去用 Javascript 实现一个 javascript 的解析器

从根本上说，JAX 是一个库，提供 API 类似 NumPy，主要用于编写的数组操纵程序进行**转换**。甚至有人认为 JAX 可以看做 Numpy v2，不仅加快 Numpy 而且为 Numpy 提供自动求导(grad)功能，让我们仅凭借 JAX 就可以去实现一个机器学习框架。

接下来主要就是来解释一下为什么说 JAX 提供 API 类似 NumPy，。现在，你可以把  JAX 看作是在加速器上运行支持自动求导的 NumPy。

```python
import jax
import jax.numpy as jnp

x = jnp.arange(10)
print(x)
```


如果大家熟悉或用过 numpy 写过点东西，上面的代码应该不会陌生，这也就是 JAX的魅力，可以从 numpy 无缝过渡到 JAX 在于你不需要学习一个新的 API。可以将以前用用 numpy 实现的代码，可以用 `jnp` 代替 `np`，程序也可以运行起来，当然也有不同之处，随后会介绍。在 `jnp` 是 DeviceArray 类型的变量，这也是 JAX 表示数组的方式。


我们现在将计算两个向量的点积，`block_until_ready` 在无需更改代码在 GPU 的设备运行代码，而不需要改变代码。使用`%timeit`来检查性能。

> 技术细节：当一个 JAX 函数被调用时，相应的操作被派发到一个加速器上，通过是进行异步计算。因此，计算返回的数组不一定在函数返回时就被“填满"。因此，如果不需要立即得到结果，因为是异步计算，所以不会阻塞 Python 的执行。因此，除非设置 block_until_ready，否则我们将只为调度计时，而不是为实际计算计时。参见 JAX 文档中的**异步调度**



```python
long_vector = jnp.arange(int(1e7))

%timeit jnp.dot(long_vector, long_vector).block_until_ready()
```



```python
The slowest run took 4.37 times longer than the fastest. This could mean that an intermediate result is being cached.
100 loops, best of 5: 6.37 ms per loop
```



### JAX 的第一次转换：grad
JAX的一个基本特征是允许**转换函数**。最常用的转换之一 是 `jax.grad`，接收一个用 Python 编写的数值函数，并返回一个新的 Python 函数，计算原函数的梯度。定义一个函数`sum_of_squares`，接收一个数组并返回对数组每个元素平方后求和。

```python
def sum_of_squares(x):
  return jnp.sum(x**2)
```



对`sum_of_squares`应用 `jax.grad`将返回一个不同的函数，这个函数就是`sum_of_squares` 相对于其第一个参数 x 的梯度。

然后，将数组输入这个求导函数来返回相对于数组中每个元素的导数。



```python
sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))

print(sum_of_squares_dx(x))
```

```python
0.0
[2. 4. 6. 8.]
```

你可以通过类比向量微积分中的  $nabla$ 运算符为 jax.grad，如果函数$f(x)$ 输入给了 `jax.grad` ，也就等同于返回 $nabla$ 函数用于计算𝑓梯度的函数。


$$
(\nabla f)(x_i) = \frac{\partial f}{\partial x_i}(x_i)
$$


类似地，`jax.grad(f)` 是计算梯度的函数，所以 `jax.grad(f)(x)`是 `f`在 `x` 处的梯度。(和$\nabla$一样，`jax.grad`只对有标量输出的函数起作用，否则会引发错误)

这样一来 JAX API 与其他支持自动求导如 Tensorflow 和 PyTorch 深度学习框架就有很大的不同，在后者中，我们可以使用损失张量本身来计算梯度( 例如通过调用 loss.backward() 来计算梯度)。JAX API 直接与函数一起工作，更接近于底层数学。一旦你习惯了这种做事方式，就会感觉很自然：你在代码中的损失函数确实是一个参数和数据的函数，你就像在数学中那样找到它的梯度。

这种做事方式使得控制诸如对哪些变量进行微分的事情变得简单明了。默认情况下，jax.grad会找到与第一个参数有关的梯度。在下面的例子中，sum_squared_error_dx的结果将是sum_squared_error相对于x的梯度。



```python
def sum_squared_error(x, y):
  return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)

y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(sum_squared_error_dx(x, y))
```

如果需要计算不同参数(或几个参数)的梯度，可以设置 argnums 来实现。

```python
[-0.20000005 -0.19999981 -0.19999981 -0.19999981]
```



```python
jax.grad(sum_squared_error, argnums=(0, 1))(x, y)  # Find gradient wrt both x & y
```

```python
(DeviceArray([-0.20000005, -0.19999981, -0.19999981, -0.19999981], dtype=float32),
 DeviceArray([0.20000005, 0.19999981, 0.19999981, 0.19999981], dtype=float32))
```



这是否意味着在进行机器学习时，模型需要用巨大的参数列表来编写函数，每个模型参数阵列都有一个参数？JAX 配备了将数组捆绑在称为 "pytrees " 的数据结构中的机制，`jax.grad`的使用是这样的。



### Value 和 Grad



```python
jax.value_and_grad(sum_squared_error)(x, y)
```

```python
(DeviceArray(0.03999995, dtype=float32),
 DeviceArray([-0.20000005, -0.19999981, -0.19999981, -0.19999981], dtype=float32))
```



### 辅助数据

除了想要记录数值之外，我们还经常想要报告在计算损失函数时获得的一些中间结果。但是如果我们试图用普通的`jax.grad`来做这个，就会遇到麻烦。

```python
def squared_error_with_aux(x, y):
  return sum_squared_error(x, y), x-y

jax.grad(squared_error_with_aux)(x, y)
```



上面代码执行会报错，还需在`grad`函数中设置一个参数。



```pyt
jax.grad(squared_error_with_aux, has_aux=True)(x, y)
```



这是因为`jax.grad`只定义在标量函数上，转换后得到函数会返回一个元组。因为组员中包含一些辅助数据， 这就是`has_aux`的作用。



### JAX 与 NumPy 不同之处

通过上面例子我们已经发现 jax.numpy 在 API 设计上基本可以说与 NumPy 的 API 保持一致。然而，并非全部也有一些的区别。接下来我们就 JAX 与 Numpy 不同之处给大家介绍一下。最重要的区别，就是 JAX 更偏向于函数式编程的风格，这是 Numpy 和 JAX 在某些点不仅相同主要原因。对函数式编程（FP）的介绍不在本指南的范围之内。如果已经熟悉了 FP，那么用起来 JAX 就会更加顺手，因为 JAX 就是面向函数式编程设计的。

```python
import numpy as np

x = np.array([1, 2, 3])

def in_place_modify(x):
  x[0] = 123
  return None

in_place_modify(x)
x
```



如果熟悉函数式编程，当看出输出`array([123,   2,   3])`时，就会发现问题了，`in_place_modify` 做了一些侧边效应的事，在其内部更新 x 的值。因为在函数式编程中数据应该是 immutable(不可变)，每次修改数据不是在源数据上进行修改，而是 copy 一份在进行修改。

```python
in_place_modify(jnp.array(x)
```



有用的是，这个错误给指出了 JAX 通过 ` jax.ops.index_* ops` 做是一个无副作用的方法。类似于不应该通过索引在原数组上进行的就地修改(in-place modification)，而是创建一个新的数组并进行相应的修改。所以上面操作在 JAX 中会报错



```python
def jax_in_place_modify(x):
  return jax.ops.index_update(x, 0, 123)

y = jnp.array([1, 2, 3])
jax_in_place_modify(y)
```



```
DeviceArray([123,   2,   3], dtype=int32)
```

这时我们再次查看 y 发现并没有改变。

```
y #DeviceArray([1, 2, 3], dtype=int32)
```



 Side-effect-free code is sometimes called *functionally pure*, or just *pure*.

>  无副作用的代码有时被称为功能上的 pure，不是功能单一意思，而是不做一些更新应用状态，或者 IO 等等其他工作。



pure 版本的效率不是更低吗？严格地说，是的。这是我们不是在原有数据进行修改而是创建一个新的数组在其上进行修改。然而，JAX 计算在运行前通常会使用另一个程序转换，即 `jax.jit` 进行编译。如果我们在使用 `jax.ops.index_update()`对原数组进行 "就地 "修改后不使用，编译器就能识别出实际上可以编译为**就地修改**，从而最终得到高效的代码。

当然，有可能将有副作用的 Python 代码和函数式支持存函数的 JAX 代码混合在一起，其实很难写出或者几乎做不到，写出纯函数式编程的程序，随着你对 JAX 越来越熟悉，就会逐渐熟练知道什么时候该用 JAX，在后面有关这一点还会调到，暂时我们就记住在 JAX 中避免发生侧边效用。



### 第一个基于 JAX 训练迭代


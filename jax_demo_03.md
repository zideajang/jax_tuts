在这一节中，我们将深究一下 JAX 是如何运行的，以及我们如何使其具有性能。将讨论`jax.jit()`转换，将对一个 JAX 的 Python 函数进行及时(JIT)编译，以便在XLA中有效地执行。



在之前有关 JAX 分享中，我们已经了解到了  JAX 可以将 Python 函数进行转换得到一个新的函数。这是通过首先将 Python 函数转换为一种叫做 jaxpr 的简单的中间语言来实现的。然后，转换在 jaxpr 的表示上工作。

接下来使用`jax.make_jaxpr`来展示一个函数的 jaxpr 表示一个 python 函数。



Conceptually, one can think of JAX transformations as first trace-specializing the Python function to be transformed into a small and well-behaved intermediate form that is then interpreted with transformation-specific interpretation rules. One of the reasons JAX can pack so much power into such a small software package is that it starts with a familiar and flexible programming interface (Python with NumPy) and it uses the actual Python interpreter to do most of the heavy lifting to distill the essence of the computation into a simple statically-typed expression language with limited higher-order features. That language is the jaxpr language.

从概念上讲，把  JAX 转换首先要做的是的 Python 函数化为一个轻量级、具有良好表现形式的中间形式，这个过程可以理解为特定的 trace，Jaxpr 经过内部解释器执行变换。JAX 能够在如此小巧的软件包中塞入这么多的功能，其中原因是不仅从一个熟悉的、灵活的编程接口（带有 NumPy 的Python）开始，并使用实际的 Python 解释器来完成大部分繁重的工作，将计算的本质提炼成一种简单的静态类型的表达式语言，具有有限的高阶特征。这种语言就是 jaxpr 语言。



```python
import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
```



```python
{ lambda  ; a.
  let b = log a
      c = log 2.0
      d = div b c
  in (d,) }
```





文档中的 "理解Jaxprs "部分提供了关于上述输出含义的更多信息。



重要的是，请注意jaxpr 并没有对函数的副作用进行 `trace`：在转换得到中 jaxpr 中没有找到`global_list.append(x)`的内容。这是一个特点，而不是一个 error 。JAX 的设计是为了理解无副作用(也就是纯函数)的代码。

JAX 内部表示是纯函数式的，但考虑到 Python 语言高度动态性特点，对用户使用上有一些编程限制。比如 JAX 自动微分的 Python 函数只支持纯函数，要求用户自行保证这一点。如用户代码写了副作用，可能经过 JAX 变换生成的函数执行结果不符合期望。因 JAX trace 函数为纯函数，当全局变量、配置信息发生变化，可能需要重新 trace。



trace 过程中，JAX 用追踪器对象(tracer object)来包裹每个参数，然后这些追踪器记录了在函数调用过程中对参数进行的所有 JAX 操作（这发生在普通的 Python 中）。然后，JAX 使用追踪器的记录来重构整个函数。这个重构的输出是就是中间的 jaxpr。因为追踪器不会记录 Python 的副作用，副作用的代码不会出现在 jaxpr 中。其中在跟踪过程中，副作用仍然发生。

```python
def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))
```



注意：Python 的 print() 函数也并非存函数，因为文本输出输入 IO 操作，可以看成副作用，所以 `print` 也并非纯函数。因此，任何print() 并不会出现在 jaxpr 中。

```python
printed x: Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
{ lambda  ; a.
  let b = log a
      c = log 2.0
      d = div b c
  in (d,) }
```

See how the printed x is a Traced object? That’s the JAX internals at work.

看到打印出来的 x 是一个跟踪对象了吗？那是 JAX 内部的工作。

The fact that the Python code runs at least once is strictly an implementation detail, and so shouldn’t be relied upon. However, it’s useful to understand as you can use it when debugging to print out intermediate values of a computation.

Python 代码至少运行一次这一事实严格来说是一个实现细节，所以不应该被依赖。然而，理解这一点很有用，因为你可以在调试时使用它来打印出计算的中间值。

A key thing to understand is that jaxpr captures the function as executed on the parameters given to it. For example, if we have a conditional, jaxpr will only know about the branch we take:

需要理解的一个关键问题是，jaxpr 捕获的是在给定参数上执行的函数。我们有一个条件，jaxpr 将只知道我们采取的分支。

```python
def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
```



```python
{ lambda  ; a.
  let 
  in (a,) }
```



上面的代码是一次向加速器发送一个操作。这限制了 XLA 编译器优化我们函数的能力。

当然，我们要做的是给XLA编译器尽可能多的代码，这样它就可以完全优化它。为此，JAX提供了jax.jit转换，它将JIT编译一个与JAX兼容的函数。下面的例子显示了如何使用JIT来加速前面的函数。
## 如何在Python中从头开始实现一个高性能和可扩展的深度神经网络？

- 硬件加速(GPU/TPU)
- 通过 autodiff 实现快速优化
- 通过融合(fusion) 来优化编译
- 向量化(vectorized batching of opertions)
- Paralleization of data and computation

```python
def predict(params, inputs):
  for W, b in params:
    outputs = np.dot(inputs, W) + b
    inputs = np.tanh(outputs)
  return outputs
```



```python
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return np.sum((preds - targets) ** 2)
```





- GPU/TPU
- autodiff
- batching
- compilation
- Parallelization



```python
import jax.numpy as jnp
from jax import grad,vmap,pmap,jit
```



```python
def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)
  return outputs
```



```python
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return jnp.sum((preds - targets) ** 2)
```



```python
gradient_fun = jit(grad(loss))
perexample_grads = jit(pmap(grad(loss),in_axes=(None,0)))
```



数值计算是关于表达和转换数学函数的。(expressing and transforming)



JAX is an extensible system for composable function transformations based on trace-specializing functional Python code.

JAX是一个可扩展的系统，用于基于跟踪专业化的功能Python代码的可组合的函数转换。



```python
from jax import random
```


# coding: utf-8

# 4.4.1 勾配法
# 最適なパラメータとは、損失関数が最小値 = 勾配を見る

import numpy as np
import matplotlib.pylab as plt

def _numerical_gradient_no_batch(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x) # x と同じ形状の配列を生成

  for idx in range(x.size):
    tmp_val = x[idx]

    x[idx] = float(tmp_val) + h
    fxh1 = f(x) # f(x+h)
    
    x[idx] = tmp_val - h 
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)
    
    x[idx] = tmp_val # 値を元に戻す
  
  return grad


def numerical_gradient(f, X):
  # 今回の計算は X が一次元
  if X.ndim == 1:
    return _numerical_gradient_no_batch(f, X)
  else:
    grad = np.zeros_like(X)
    
    for idx, x in enumerate(X): # enumerate インデックス付き要素を得る
        grad[idx] = _numerical_gradient_no_batch(f, x)
    
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x
  x_history = []

  for i in range(step_num):
    x_history.append( x.copy() )
    grad = numerical_gradient(f, x) # 勾配
    x = x - lr * grad # 勾配に学習率を掛けた値で更新し続ける

  return np.array(x_history)


def function_2(x):
  return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0]) # 初期座標
lr = 0.1 # 学習率
step_num = 20 # 勾配法の繰り返し数
x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0,0], '--b') # b は blue
plt.plot([0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o') # 散布図
print(x_history) # function_2 が表現できる最小値 0 に漸近する

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

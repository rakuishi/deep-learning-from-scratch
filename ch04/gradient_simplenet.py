# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


def _loss(w, x, t):
  # 入力層 → 出力層
  # 計算結果 y の最大値のインデックスを正解値にしたいが、ガウス分布の出来による
  a1 = np.dot(x, w) # (2,) * (2,3) = (3,)
  y  = softmax(a1) # y.shape = (3,), y.ndim = 1, y.size = 3
  # print('y :' + str(y))

  # 出力 y と教師 t を比較して損失を得る（交差エントロピー誤差を用いる）
  loss = cross_entropy_error(y, t)
  # print('l : ' + str(loss))
  return loss


def loss(dummy):
  return _loss(w, x, t)


w = np.random.randn(2,3) # ガウス分布で正規化 (2,3)
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
print('W : ' + str(w)) # 重みパラメータ W

# w.shape(2,3) の各インデックス毎に h を変え、
# gradient（勾配）= (f(x+h) - f(x-h)) / (2*h)
dW = numerical_gradient(loss, w)
print('dW: ' + str(dW)) # ニューラルネットワークにおける勾配 dW

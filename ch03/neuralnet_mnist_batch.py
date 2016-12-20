# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize = True, one_hot_label = False)
  return x_test, t_test


def init_network():
  # 学習済みの重みパラメータ（隠れ層はふたつある）を読み込む
  with open('sample_weight.pkl', 'rb') as f:
    network = pickle.load(f)
  return network


def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y

# 3.6.2 ニューラルネットワークの推論処理では、10,000 枚のテスト画像を 1 枚ずつ処理していたが、
# 3.6.3 バッチ処理では、100 枚ずつバッチ処理を行う

x, t = get_data()
network = init_network()

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # 0, 100, 200, ... , 10,000
  x_batch = x[i:i+batch_size] # [0:100], [100:200], ... , [9900:10000]
  y_batch = predict(network, x_batch) # y_batch.shape = (100, 10)
  p = np.argmax(y_batch, axis=1) # p.shape = (100,)
  accuracy_cnt += np.sum(p == t[i:i+batch_size])

# この学習済みのニューラルネットワークの正解精度は 0.9352%
print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))

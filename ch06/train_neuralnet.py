# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num       = 10000 # 繰り返し数
train_size      = x_train.shape[0] # 訓練データ総数 60,000
batch_size      = 100 # バッチ
learning_rate   = 0.1 # 学習率
train_loss_list = []
train_acc_list  = []
test_acc_list   = []
iter_per_epoch  = max(train_size / batch_size, 1) # 600

for i in range(iters_num):
  # すべての訓練データから 100 個をランダムに抽出して、訓練データと教師データを取る
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # 勾配を計算する
  grad = network.gradient(x_batch, t_batch)

  # パラメータを更新する（勾配法）
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] = network.params[key] - learning_rate * grad[key]

  # 損失関数の値を記録用に求める
  # loss = network.loss(x_batch, t_batch)
  # train_loss_list.append(loss)

  # 10000 / 600 回ぐらいの割合で訓練画像とテスト画像に対する認識精度を確認する
  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print('train acc, test acc | ' + str(train_acc) + ", " + str(test_acc))

# グラフ描画
# テスト用画像では、0.945 程度の認識率
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

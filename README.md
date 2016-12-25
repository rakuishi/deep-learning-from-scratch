# ゼロから作る Deep Learning

本リポジトリでは、オライリー・ジャパン発行書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』を写経しています。

## 環境構築

### 1.6.1 単純なグラフの描画

```
$ python matplotlib-sin.py
Matplotlib is building the font cache using fc-list.
$ brew install fontconfig
$ fc-list
```

## 認識精度

### 4.5 ニューラルネットワーク

`cd ch04 && python train_neuralnet.py`

- 認識精度: 94.5%
- 勾配の計算方法: 誤差逆伝播法

<img src="https://raw.githubusercontent.com/rakuishi/deep-learning-from-scratch/master/ch04/train_neuralnet.png" width="400">

### 6.2 ニューラルネットワーク（最適化）

`cd ch06 && python train_neuralnet.py`

- 認識精度: 96.5%（訓練データに対しては 97.9%）
- 学習率: 0.1 → 0.001
- 更新手法: 確率的勾配降下法（stochastic gradient descent, SGD） → Adam
- 重みの初期値: ガウス分布 → Xavier 初期値

<img src="https://raw.githubusercontent.com/rakuishi/deep-learning-from-scratch/master/ch06/train_neuralnet.png" width="400">

## 参考

* [ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)
* [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)

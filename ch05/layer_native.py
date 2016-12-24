# coding: utf-8


# 乗算レイヤ
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    return x * y

  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    return dx, dy


# 可算レイヤ
class AddLayer:
  def __init__(self):
    pass

  def forward(self, x, y):
    return x + y

  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy

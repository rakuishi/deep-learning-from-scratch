# coding: utf-8
from layer_native import *


apple      = 100
apple_num  = 2
orange     = 150
orange_num = 3
tax        = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(apple_orange_price, tax)

# backward
d_price = 1
d_apple_orange_price, d_tax = mul_tax_layer.backward(d_price)
d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_apple_orange_price)
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)
d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)

print('price:', int(price))
print('d_apple: ', d_apple)
print('d_apple_num: ', int(d_apple_num))
print('d_orange: ', d_orange)
print('d_orange_num: ', int(d_orange_num))
print('d_apple_orange_price:', d_apple_orange_price)
print('d_tax:', d_tax)

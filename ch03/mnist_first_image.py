# coding: utf-8

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def img_show(img, size):
  img = img.reshape(size, size)
  pil_image = Image.fromarray(np.uint8(img))
  pil_image.show()

(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten = True, normalize = False)

img = x_train[0]
img_show(img, 28)

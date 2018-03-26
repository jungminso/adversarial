### FGSM code from [gongzhitaao's github](https://github.com/gongzhitaao/tensorflow-adversarial)

```python
"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from fast_gradient import fgm


img_size = 28
img_chan = 1
n_classes = 10
```

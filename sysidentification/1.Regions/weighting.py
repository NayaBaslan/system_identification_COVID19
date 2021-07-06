from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sird_model import sird_model_params
import os
import pdb
from plots_utils import *

import xlsxwriter
N1 = 330
W = np.ones((4,N1))
for i in range(4):
    for j in range(N1):
        W[i,j] = (j+1)*0.1

tspan = np.arange(0, N1, 1)
max_abs = np.max(np.abs(W))
plt.imshow(W, cmap='RdBu', vmin=-max_abs, vmax=max_abs)

pdb.set_trace()
plt.show()
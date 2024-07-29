import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from scipy.optimize import fsolve

model = nn.BatchNorm1d(num_features=4, eps=0, momentum=0)

input = torch.randn(1200, 4)

output = model(input)

mean = torch.mean(output, dim=0)
std = torch.std(output, dim=0)
cov = torch.cov(output.T)

print("mean = {}".format(mean))
print("std = {}".format(std))
print("cov matrix = {}".format(cov))
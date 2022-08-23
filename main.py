import torch

from model import *

my_model = MAXIM(num_supervision_scales=3, features=32)
input_test = torch.ones((1, 64, 64, 3))
y = my_model(input_test)

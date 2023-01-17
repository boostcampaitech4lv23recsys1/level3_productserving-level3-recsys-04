import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class trash_model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, user, rest):
        return random.random()
# model = trash_model()
# print(model(1,2) # user_id, rest_id) => output(0~1) 나오는 모델
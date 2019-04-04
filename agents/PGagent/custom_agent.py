import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import textworld
from textworld import EnvInfos

class CustomAgent:
    def __init__(self):
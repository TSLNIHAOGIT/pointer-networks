#encoding=utf8
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras import backend as K
import tensorflow as tf

batch_size = 1
time_step = 3
dim = 2
x = np.random.rand(batch_size, time_step, dim)  # [1,3,2]生成输入
init_state = np.zeros(1).reshape(1, 1)  # [1,1] 初始值设置为0

def step_func(inputs, states):
    o = K.sum(inputs, axis=1, keepdims=True) + states[0]
    return o, [o]

a,_,_ = K.rnn(step_func, inputs=x, initial_states=[init_state])
print("x", x)

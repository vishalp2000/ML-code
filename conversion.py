import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from collections import OrderedDict
import tensorflow as tf

import model as m
shape = 20

device = 'cpu'

trained_model = m.NeuralNetwork(shape).to(device)
trained_model.load_state_dict(torch.load('models/805_08-13-12-model.pth', map_location=torch.device(device)))

dummy_input = Variable(torch.randn(1, 1, shape, shape)) 
torch.onnx.export(trained_model, dummy_input, "models/805.onnx")

#  Load the ONNX file
model = onnx.load('models/805.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

tf_rep.export_graph('models/805.pb')

converter = tf.lite.TFLiteConverter.from_saved_model("models/805.pb")
tflite_model = converter.convert()
open("models/805.tflite", "wb").write(tflite_model)
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
trained_model.load_state_dict(torch.load('C:/Users/visha/models1/502_12-42-26-model.pth', map_location=torch.device(device)))
#"C:\Users\visha\models1\502_12-42-26-model.pth"
#models/805_08-13-12-model.pth
dummy_input = Variable(torch.randn(1, 1, shape, shape)) 
torch.onnx.export(trained_model, dummy_input, "models1/vishal.onnx")

#  Load the ONNX file
model = onnx.load('models1/vishal.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

tf_rep.export_graph('models1/vishal.pb')

converter = tf.lite.TFLiteConverter.from_saved_model("models1/vishal.pb")
tflite_model = converter.convert()
open("models1/vishal.tflite", "wb").write(tflite_model)

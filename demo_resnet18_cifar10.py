from deepview import DeepView
import numpy as np
import time
import torch
import sys
import datetime
import os
import evaluate
# ---------------------------
import demo_utils as demo

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "../../DVI_EXP/normal_training/resnet18_cifar10"
sys.path.append(content_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print(device)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# data = torch.load("../../DVI_EXP/normal_training/resnet18_cifar10/Training_data/training_dataset_data.pth").cpu().numpy()
# targets = torch.load("../../DVI_EXP/normal_training/resnet18_cifar10/Training_data/training_dataset_label.pth").cpu().numpy()
data = torch.load("E:\\DVI_exp_data\\resnet18_cifar10\\Training_data\\training_dataset_data.pth").cpu().numpy()
targets = torch.load("E:\\DVI_exp_data\\resnet18_cifar10\\Training_data\\training_dataset_label.pth").cpu().numpy()

# model_location = "../../DVI_EXP/normal_training/resnet18_cifar10/Model/Epoch_200/subject_model.pth"
model_location = "E:\\DVI_exp_data\\resnet18_cifar10\\Model/Epoch_200\\subject_model.pth"
net.load_state_dict(torch.load(model_location, map_location=device))
net.to(device)
net.eval()

softmax = torch.nn.Softmax(dim=-1)
def pred_wrapper(x):
    with torch.no_grad():
        x = np.array(x, dtype=np.float32)
        tensor = torch.from_numpy(x).to(device)
        logits = net(tensor)
        probabilities = softmax(logits).cpu().numpy()
    return probabilities

# --- Deep View Parameters ----
batch_size = 128
max_samples = 100000
data_shape = (3, 32, 32)
n = 5
lam = .65
resolution = 100
cmap = 'tab10'
title = 'ResNet-18 - CIFAR10'


deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                    data_shape, n, lam, resolution, cmap, title=title)

t0 = time.time()
deepview.add_samples(data[:100], targets[:100])
t1 = time.time()
print((t1-t0))


# pick samples for training and testing
train_samples = deepview.samples
train_embeded = deepview.embedded
train_pred = deepview.y_pred
train_labels = deepview.y_true
train_recon = deepview.inverse(train_embeded)
print(evaluate.evaluate_proj_nn_perseverance_knn(train_samples.reshape(100,-1), train_embeded, 10))
print(evaluate.evaluate_inv_nn(train_samples.reshape(100,-1), train_recon.reshape(100, -1), 10))
print(evaluate.evaluate_inv_accu(train_labels, train_pred))
ori_pred = deepview.predict_batches(train_samples)
new_pred = deepview.predict_batches(train_recon)
print(evaluate.evaluate_inv_conf(train_pred.astype(np.int), ori_pred, new_pred))


# boundary preserving
train_samples = deepview.samples
train_embeded = deepview.embedded

print(evaluate.evaluate_proj_boundary_perseverance_knn(train_samples[:50].reshape(50, -1), train_embeded[:50], train_samples[50:].reshape(50, -1), train_embeded[50:], 15))



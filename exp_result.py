"""
This is the experiment for baseline umap on nn_preserving, boundary_preserving, inv_preserving, inv_accu, inv_conf_diff, and time
"""
import torch
from deepview import DeepView
import os
import argparse
import evaluate
import sys
import numpy as np
import utils
import time
import json
import random
from scipy.special import softmax


def main(args):
    result = list()

    num = args.train_num
    train_num = args.train_num

    if args.dataset == "CIFAR10":
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    elif args.dataset == "MNIST":
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    else:
        classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    content_path = args.content_path
    sys.path.append(content_path)
    from Model.model import resnet18
    net = resnet18()

    epoch_id = args.epoch_id
    device = torch.device(args.device)

    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

    net.load_state_dict(torch.load(model_location, map_location=device))
    net.to(device)
    net.eval()

    train_data = np.load(os.path.join(content_path,"Model", "Epoch_{:d}".format(epoch_id), "dv_train_data.npy"))
    train_embedding = np.load(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_embedding.npy"))
    train_recon = np.load(os.path.join(content_path,"Model", "Epoch_{:d}".format(epoch_id), "dv_train_recon.npy"))
    train_pred = np.load(os.path.join(content_path,"Model", "Epoch_{:d}".format(epoch_id), "dv_train_pred.npy"))
    train_labels = np.load(os.path.join(content_path,"Model", "Epoch_{:d}".format(epoch_id), "dv_train_labels.npy"))

    #
    result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 15))
    result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 20))
    result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 30))

    result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=15))
    result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=20))
    result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=30))

    ori_pred = softmax(utils.batch_run(net, torch.from_numpy(train_data), 10),axis=1)
    new_pred = softmax(utils.batch_run(net, torch.from_numpy(train_recon), 10), axis=1)
    result.append(evaluate.evaluate_inv_accu(train_labels, train_pred))
    result.append(evaluate.evaluate_inv_conf(train_pred.astype(np.int), ori_pred, new_pred))

    # boundary preserving

    fitting_data = np.load(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_fitting_data.npy"))
    fitting_embedding = np.load(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "dv_fitting_embedding.npy"))

    # boundary preserving
    border_num = int(train_num/10)

    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(fitting_data[:train_num].reshape(train_num, -1),
                                                                   fitting_embedding[:train_num],
                                                                   fitting_data[train_num:].reshape(border_num, -1),
                                                                   fitting_embedding[train_num:], 15))
    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(fitting_data[:train_num].reshape(train_num, -1),
                                                                   fitting_embedding[:train_num],
                                                                   fitting_data[train_num:].reshape(border_num, -1),
                                                                   fitting_embedding[train_num:], 20))
    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(fitting_data[:train_num].reshape(train_num, -1),
                                                                   fitting_embedding[:train_num],
                                                                   fitting_data[train_num:].reshape(border_num, -1),
                                                                   fitting_embedding[train_num:], 30))

    with open(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "deepview_{:d}_exp_result.json".format(epoch_id)), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_id", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--advance_attack", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_shape", nargs='+', type=int)
    parser.add_argument("--train_num", type=int)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "MNIST", "FASHIONMNIST"])
    args = parser.parse_args()
    main(args)







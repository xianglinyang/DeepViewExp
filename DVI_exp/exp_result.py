"""
This is the experiment for baseline umap on nn_preserving, boundary_preserving, inv_preserving, inv_accu, inv_conf_diff, and time
"""
import torch
import os
import argparse
import evaluate
import sys
import numpy as np
import utils
import json
from scipy.special import softmax


def main(args):
    result = list()

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

    input_name = args.dataset + "_" + epoch_id
    input_dir = os.path.join(".", "results", input_name)

    discr_distances = np.load(os.path.join(input_dir, "fisher_dist.npy"))
    eucl_distances = np.load(os.path.join(input_dir, "eucli_dist.npy"))
    train_embedding = np.load(os.path.join(input_dir, "train_embedding.npy"))
    border_embedding = np.load(os.path.join(input_dir, "border_embedding.npy"))
    train_recon = np.load(os.path.join(input_dir, "train_recon.npy"))
    train_data = np.load(os.path.join(input_dir, "train_data.npy"))

    # based on paper hyperparameters selection, lambda = .65
    dists = .65 * eucl_distances + .35 * discr_distances
    train_num = train_embedding.shape[0]


    result.append(evaluate.evaluate_proj_nn_perseverance_knn(dists, train_num, train_embedding, 15))
    result.append(evaluate.evaluate_proj_nn_perseverance_knn(dists, train_num, train_embedding, 20))
    result.append(evaluate.evaluate_proj_nn_perseverance_knn(dists, train_num, train_embedding, 30))

    # boundary preserving
    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(dists, train_num, train_embedding, border_embedding, 15))
    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(dists, train_num, train_embedding, border_embedding, 20))
    result.append(evaluate.evaluate_proj_boundary_perseverance_knn(dists, train_num, train_embedding, border_embedding, 30))

    # reconstruction confidence
    ori_pred = softmax(utils.batch_run(net, torch.from_numpy(train_data), 10), axis=1)
    new_pred = softmax(utils.batch_run(net, torch.from_numpy(train_recon), 10), axis=1)
    ori_pred = ori_pred.argmax(1)
    recon_pred = new_pred.argmax(1)
    result.append(evaluate.evaluate_inv_accu(ori_pred, recon_pred))
    result.append(evaluate.evaluate_inv_conf(ori_pred.astype(np.int), ori_pred, new_pred))

    with open(os.path.join(input_dir, "exp_result.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_id", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_shape", nargs='+', type=int)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "MNIST", "FASHIONMNIST"])
    args = parser.parse_args()
    main(args)

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

    train_path = os.path.join(content_path, "Training_data")
    train_data = torch.load(os.path.join(train_path, "training_dataset_data.pth")).cpu().numpy()
    train_label = torch.load(os.path.join(train_path, "training_dataset_label.pth")).cpu().numpy()

    num_list = np.arange(len(train_data))
    index = random.sample(num_list.tolist(), args.train_num)

    train_data = train_data[index]
    train_label = train_label[index]

    # if args.advance_attack == 0:
    #     border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "ori_border_centers.npy")
    #     border_points = np.load(border_points)
    # else:
    #     border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "ori_advance_border_centers.npy")
    #     border_points = np.load(border_points)[:100]
    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

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
    batch_size = 200
    max_samples = 100000
    data_shape = tuple(args.data_shape)
    n = 5
    lam = .65
    resolution = 100
    cmap = 'tab10'
    title = 'ResNet-18 - CIFAR10'

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title)

    t0 = time.time()
    deepview.add_samples(train_data, train_label)
    t1 = time.time()
    print(t1 - t0)
    # training time
    result.append(round(t1-t0, 4))

    # pick samples for training and testing
    train_samples = deepview.samples
    train_embedding = deepview.embedded
    train_pred = deepview.y_pred
    train_labels = deepview.y_true
    train_recon = deepview.inverse(train_embedding)
    num = len(index)

    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_data.npy"), train_samples)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_embedding.npy"), train_embedding)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_recon.npy"), train_recon)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_pred.npy"), train_pred)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"dv_train_labels.npy"), train_labels)

    #
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 10))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 15))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data.reshape(num, -1), train_embedding, 30))
    #
    # result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=10))
    # result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=15))
    # result.append(evaluate.evaluate_inv_nn(train_data.reshape(num, -1), train_recon.reshape(num, -1), n_neighbors=30))
    #
    # ori_pred = deepview.predict_batches(train_samples)
    # new_pred = deepview.predict_batches(train_recon)
    # result.append(evaluate.evaluate_inv_accu(train_labels, train_pred))
    # result.append(evaluate.evaluate_inv_conf(train_pred.astype(np.int), ori_pred, new_pred))

    # # boundary preserving
    # train_samples = deepview.samples
    # train_embeded = deepview.embedded
    #
    # print(evaluate.evaluate_proj_boundary_perseverance_knn(train_samples[:50].reshape(50, -1), train_embeded[:50],
    #                                                        train_samples[50:].reshape(50, -1), train_embeded[50:], 15))


    #
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      10))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      15))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      30))
    #
    # with open(os.path.join(content_path, "deepview_exp_result.json"), "w") as f:
    #     json.dump(result, f)


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







"""
This is the experiment for baseline umap on nn_preserving, boundary_preserving, inv_preserving, inv_accu, inv_conf_diff, and time
"""
import torch
from deepview import DeepView
from deepview.fisher_metric import calculate_fisher
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

    with open(os.path.join(content_path, "index.json"), 'r') as f:
        idxs = json.load(f)

    train_data = train_data[idxs]
    train_label = train_label[idxs]

    border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "ori_advance_border_centers.npy")
    border_points = np.load(border_points)
    border_cls = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "advance_border_labels.npy.npy")
    border_cls = np.load(border_cls)

    train_num = train_data.shape[0]

    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

    fitting_points = np.vstack((train_data, border_points))
    fitting_labels = np.vstack((train_label, border_cls))

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
    resolution = 10
    cmap = 'tab10'
    title = ''

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title)

    t0 = time.time()
    Y_probs = deepview._predict_batches(fitting_points)
    Y_preds = Y_probs.argmax(axis=1)
    deepview.queue_samples(fitting_points, fitting_labels, Y_preds)
    new_discr, new_eucl = calculate_fisher(deepview.model, fitting_points, fitting_points,
                                           deepview.n, deepview.batch_size, deepview.n_classes, deepview.verbose)
    deepview.discr_distances = deepview.update_matrix(deepview.discr_distances, new_discr)
    deepview.eucl_distances = deepview.update_matrix(deepview.eucl_distances, new_eucl)
    t1 = time.time()
    deepview.mapper.fit(deepview.distances)
    t2 = time.time()
    deepview.embedded = deepview.mapper.transform(deepview.distances)
    t3 = time.time()
    deepview.inverse.fit(deepview.embedded, deepview.samples)
    t4 = time.time()
    fitting_recon = deepview.inverse(deepview.embedded)
    t5 = time.time()
    record_time = {"distance_cal": t1-t0, "proj_fit": t2-t1, "transform": t3-t2,
                   "inverse_fit": t4-t3, "inverse_transform": t5-t4}


    fitting_embedding = deepview.embedded

    output_name = args.dataset + "_" + epoch_id

    output_dir = os.path.join(".", "results", output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.save(os.path.join(output_dir, "fisher_dist.npy"), deepview.discr_distances)
    np.save(os.path.join(output_dir, "eucli_dist.npy"), deepview. eucl_distances)
    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    np.save(os.path.join(output_dir, "train_embedding.npy"), fitting_embedding[:train_num])
    np.save(os.path.join(output_dir, "border_embedding.npy"), fitting_embedding[train_num:])
    np.save(os.path.join(output_dir, "train_recon.npy"), fitting_recon[:train_num])

    with open(os.path.join(output_dir, "time.json"), "w") as f:
        json.dump(record_time, f)


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







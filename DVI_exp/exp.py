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
    if args.dim == 512:
        from Model.model import resnet18
        net = resnet18()
    else:
        from Model.model import resnet50
        net = resnet50()

    epoch_id = args.epoch_id
    device = torch.device(args.device)
    exp_round = args.exp

    train_path = os.path.join(content_path, "Training_data")
    train_data = torch.load(os.path.join(train_path, "training_dataset_data.pth")).cpu().numpy()
    train_label = torch.load(os.path.join(train_path, "training_dataset_label.pth")).cpu().numpy()

    test_path = os.path.join(content_path, "Testing_data")
    test_data = torch.load(os.path.join(test_path, "testing_dataset_data.pth")).cpu().numpy()
    test_label = torch.load(os.path.join(test_path, "testing_dataset_label.pth")).cpu().numpy()

    ## index reading
    index_dir = os.path.join(content_path, "Model", "Epoch_{}".format(epoch_id), "index.json")
    with open(index_dir, 'r') as f:
        idxs = json.load(f)
    test_index_dir = os.path.join(content_path, "Model", "Epoch_{}".format(epoch_id), "test_index.json")
    with open(test_index_dir, 'r') as f:
        test_idxs = json.load(f)

    exp_path = os.path.join(".", "batch_run_results", "{}".format(exp_round))
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    output_name = args.output_name + "_" + str(epoch_id)
    output_dir = os.path.join(exp_path, output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_data = train_data[idxs]
    train_label = train_label[idxs]
    test_data = test_data[test_idxs]
    test_label = test_label[test_idxs]

    border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "ori_advance_border_centers.npy")
    border_points = np.load(border_points)
    border_cls = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "advance_border_labels.npy")
    border_cls = np.load(border_cls)

    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

    fitting_points = np.concatenate((train_data, border_points), axis=0)
    fitting_labels = np.concatenate((train_label, border_cls), axis=0)

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
    batch_size = 500
    max_samples = 100000
    data_shape = tuple(args.data_shape)
    n = 5
    lam = .65
    resolution = 10
    cmap = 'tab10'
    title = ''

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title)
    # load from checkpoint
    if args.ckpt:
        input_dir = output_dir
        discr_distances = np.load(os.path.join(input_dir, "fisher_dist.npy"))
        eucl_distances = np.load(os.path.join(input_dir, "eucli_dist.npy"))
        train_embedding = np.load(os.path.join(input_dir, "train_embedding.npy"))
        border_embedding = np.load(os.path.join(input_dir, "border_embedding.npy"))

        with open(os.path.join(output_dir, "time.json"), "r") as f:
            record_time = json.load(f)

        Y_probs = deepview._predict_batches(fitting_points)
        Y_preds = Y_probs.argmax(axis=1)
        deepview.queue_samples(fitting_points, fitting_labels, Y_preds)

        deepview.discr_distances = deepview.update_matrix(deepview.discr_distances, discr_distances)
        deepview.eucl_distances = deepview.update_matrix(deepview.eucl_distances, eucl_distances)
        deepview.embedded = np.concatenate((train_embedding, border_embedding), axis=0)
        np.save(os.path.join(output_dir, "dist.npy"), deepview.distances)
    else:
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

        np.save(os.path.join(output_dir, "fisher_dist.npy"), deepview.discr_distances)
        np.save(os.path.join(output_dir, "eucli_dist.npy"), deepview.eucl_distances)
        np.save(os.path.join(output_dir, "dist.npy"), deepview.distances)
        np.save(os.path.join(output_dir, "train_data.npy"), train_data)
        np.save(os.path.join(output_dir, "train_embedding.npy"), fitting_embedding[:train_num])
        np.save(os.path.join(output_dir, "border_embedding.npy"), fitting_embedding[train_num:])
        np.save(os.path.join(output_dir, "train_recon.npy"), fitting_recon[:train_num])

    ## testing
    t0 = time.time()
    Y_probs = deepview._predict_batches(test_data)
    Y_preds = Y_probs.argmax(axis=1)
    deepview.queue_samples(test_data, test_label, Y_preds)
    new_discr, new_eucl = calculate_fisher(deepview.model, test_data, deepview.samples,
                                           deepview.n, deepview.batch_size, deepview.n_classes, deepview.verbose)
    print(new_eucl.shape)
    deepview.discr_distances = deepview.update_matrix(deepview.discr_distances, new_discr)
    deepview.eucl_distances = deepview.update_matrix(deepview.eucl_distances, new_eucl)
    t1 = time.time()
    record_time["test_distance_cal"] = t1-t0
    deepview.mapper.fit(deepview.distances)
    t2 = time.time()
    record_time["test_proj_fit"] = t2-t1
    deepview.embedded = deepview.mapper.transform(deepview.distances)
    t3 = time.time()
    record_time["test_transform"] = t3-t2
    deepview.inverse.fit(deepview.embedded, deepview.samples)
    t4 = time.time()
    record_time["test_inverse_fit"] = t4-t3
    fitting_recon = deepview.inverse(deepview.embedded)
    t5 = time.time()
    record_time["test_inverse_transform"] = t5-t4
    record_time["test_inverse"] = t5-t3
    record_time["test_proj"] = t3-t0
    record_time["test_len"] = test_num

    fitting_embedding = deepview.embedded

    np.save(os.path.join(output_dir, "test_fisher_dist.npy"), deepview.discr_distances)
    np.save(os.path.join(output_dir, "test_eucli_dist.npy"), deepview.eucl_distances)
    np.save(os.path.join(output_dir, "test_dist.npy"), deepview.distances)
    np.save(os.path.join(output_dir, "test_data.npy"), test_data)
    np.save(os.path.join(output_dir, "test_train_embedding.npy"), fitting_embedding[:train_num])
    np.save(os.path.join(output_dir, "test_border_embedding.npy"), fitting_embedding[train_num:-test_num])
    np.save(os.path.join(output_dir, "test_test_embedding.npy"), fitting_embedding[-test_num:])
    np.save(os.path.join(output_dir, "test_train_recon.npy"), fitting_recon[:train_num])
    np.save(os.path.join(output_dir, "test_test_recon.npy"), fitting_recon[-test_num:])
    np.save(os.path.join(output_dir, "test_border_recon.npy"), fitting_recon[train_num:-test_num])

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
    parser.add_argument("--exp", type=int)
    parser.add_argument("--ckpt", type=bool, default=True, help="whether to load from ckpt or not")
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    main(args)







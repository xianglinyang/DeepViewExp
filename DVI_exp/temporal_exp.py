"""
This is the experiment for baseline DeepView on temporal_preserving
init mapper each time with previous embedding, to preserve temporal property
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
from deepview.embeddings import init_umap


def main(args):
    # dummy classes labels
    classes = range(10)

    DIM = args.dim
    CONTENT_PATH = args.content_path
    DEVICE = torch.device(args.device)
    EXP_ROUND = args.exp
    START = args.s
    END = args.e
    PERIOD = args.p

    # prepare settings
    sys.path.append(CONTENT_PATH)
    if DIM == 2048:
        from Model.model import resnet50
        net = resnet50()
    else:
        from Model.model import resnet18
        net = resnet18()


    train_path = os.path.join(CONTENT_PATH, "Training_data")
    training_data = torch.load(os.path.join(train_path, "training_dataset_data.pth")).cpu().numpy()
    training_label = torch.load(os.path.join(train_path, "training_dataset_label.pth")).cpu().numpy()

    test_path = os.path.join(CONTENT_PATH, "Testing_data")
    testing_data = torch.load(os.path.join(test_path, "testing_dataset_data.pth")).cpu().numpy()
    testing_label = torch.load(os.path.join(test_path, "testing_dataset_label.pth")).cpu().numpy()

    for epoch in range(START, END+1, PERIOD):
        ## index reading
        index_dir = os.path.join(CONTENT_PATH, "Model", "Epoch_{}".format(epoch), "index.json")
        with open(index_dir, 'r') as f:
            idxs = json.load(f)
        test_index_dir = os.path.join(CONTENT_PATH, "Model", "Epoch_{}".format(epoch), "test_index.json")
        with open(test_index_dir, 'r') as f:
            test_idxs = json.load(f)

        exp_path = os.path.join(".", "batch_run_results", "{}".format(EXP_ROUND), "temporal")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        output_name = args.output_name + "_" + str(epoch)
        output_dir = os.path.join(exp_path, output_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        train_data = training_data[idxs]
        train_label = training_label[idxs]
        test_data = testing_data[test_idxs]
        test_label = testing_label[test_idxs]

        border_points = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "ori_advance_border_centers.npy")
        border_points = np.load(border_points)
        border_cls = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "advance_border_labels.npy")
        border_cls = np.load(border_cls)

        train_num = train_data.shape[0]
        test_num = test_data.shape[0]

        model_location = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "subject_model.pth")

        fitting_points = np.concatenate((train_data, border_points), axis=0)
        fitting_labels = np.concatenate((train_label, border_cls), axis=0)

        net.load_state_dict(torch.load(model_location, map_location=DEVICE))
        net.to(DEVICE)
        net.eval()

        softmax = torch.nn.Softmax(dim=-1)

        def pred_wrapper(x):
            with torch.no_grad():
                x = np.array(x, dtype=np.float32)
                tensor = torch.from_numpy(x).to(DEVICE)
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

        Y_probs = deepview._predict_batches(fitting_points)
        Y_preds = Y_probs.argmax(axis=1)
        deepview.queue_samples(fitting_points, fitting_labels, Y_preds)
        new_discr, new_eucl = calculate_fisher(deepview.model, fitting_points, fitting_points,
                                               deepview.n, deepview.batch_size, deepview.n_classes, deepview.verbose)
        deepview.discr_distances = deepview.update_matrix(deepview.discr_distances, new_discr)
        deepview.eucl_distances = deepview.update_matrix(deepview.eucl_distances, new_eucl)


        if epoch == START:
            deepview.mapper.fit(deepview.distances)
            deepview.embedded = deepview.mapper.transform(deepview.distances)
            fitting_embedding = deepview.embedded
        else:
            prev_embedding = os.path.join(exp_path, args.output_name + "_" + str(epoch-PERIOD))
            prev_embedding = np.load(prev_embedding)
            kwargs = {"init": prev_embedding}
            deepview.mapper = init_umap(kwargs)
            deepview.mapper.fit(deepview.distances)
            deepview.embedded = deepview.mapper.transform(deepview.distances)
            fitting_embedding = deepview.embedded
        np.save(os.path.join(output_dir, "fisher_dist.npy"), deepview.discr_distances)
        np.save(os.path.join(output_dir, "eucli_dist.npy"), deepview.eucl_distances)
        np.save(os.path.join(output_dir, "dist.npy"), deepview.distances)
        np.save(os.path.join(output_dir, "train_data.npy"), train_data)
        np.save(os.path.join(output_dir, "embedding.npy"), fitting_embedding)

        # # add samples
        # Y_probs = deepview._predict_batches(test_data)
        # Y_preds = Y_probs.argmax(axis=1)
        # deepview.queue_samples(test_data, test_label, Y_preds)
        # new_discr, new_eucl = calculate_fisher(deepview.model, test_data, deepview.samples,
        #                                        deepview.n, deepview.batch_size, deepview.n_classes, deepview.verbose)
        # deepview.discr_distances = deepview.update_matrix(deepview.discr_distances, new_discr)
        # deepview.eucl_distances = deepview.update_matrix(deepview.eucl_distances, new_eucl)
        # deepview.mapper.fit(deepview.distances)
        # deepview.embedded = deepview.mapper.transform(deepview.distances)
        # fitting_embedding = deepview.embedded
        #
        # np.save(os.path.join(output_dir, "test_fisher_dist.npy"), deepview.discr_distances)
        # np.save(os.path.join(output_dir, "test_eucli_dist.npy"), deepview.eucl_distances)
        # np.save(os.path.join(output_dir, "test_dist.npy"), deepview.distances)
        # np.save(os.path.join(output_dir, "test_data.npy"), test_data)
        # np.save(os.path.join(output_dir, "test_train_embedding.npy"), fitting_embedding[:train_num])
        # np.save(os.path.join(output_dir, "test_border_embedding.npy"), fitting_embedding[train_num:-test_num])
        # np.save(os.path.join(output_dir, "test_test_embedding.npy"), fitting_embedding[-test_num:])


def temporal_preserving_train(args, n_neighbors):
    """evalute training temporal preserving property"""
    DATASET = args.dataset
    START = args.s
    END = args.e
    PERIOD = args.p
    EXP_ROUND = args.exp
    OUTPUT_PATH = os.path.join(".", "batch_run_results", "{}".format(EXP_ROUND), "temporal")

    l = 1000
    eval_num = int((END - START) / PERIOD)
    alpha = np.zeros((eval_num, l))
    delta_x = np.zeros((eval_num, l))
    for epoch in range(START+PERIOD, END+1, PERIOD):
        prev_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch-PERIOD))
        prev_embedding = np.load(os.path.join(prev_dir, "embedding.npy"))[:l]
        prev_data = np.load(os.path.join(prev_dir, "dist.npy"))[:l, :l]
        curr_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch))
        embedding = np.load(os.path.join(curr_dir, "embedding.npy"))[:l]
        curr_data = np.load(os.path.join(curr_dir, "dist.npy"))[:l, :l]

        alpha_ = evaluate.find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors)
        delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)

        alpha[int((epoch - START) / PERIOD - 1)] = alpha_
        delta_x[int((epoch - START) / PERIOD - 1)] = delta_x_

    val_corr = evaluate.evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
    return val_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_shape", nargs='+', type=int)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "MNIST", "FASHIONMNIST"])
    parser.add_argument("--dim", type=int)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--exp", type=int)
    parser.add_argument("-s", type=int)
    parser.add_argument("-e", type=int)
    parser.add_argument("-p", type=int)
    args = parser.parse_args()
    main(args)
    print(temporal_preserving_train(args, n_neighbors=15))








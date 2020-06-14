from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, device, dataloader):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    overall_total = 0
    overall_correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for batch_i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.float()

        with torch.no_grad():
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs))
        #             print(outputs)
        #             print(targets)
        total = 0
        correct = 0
        for id in range(targets.shape[0]):
            for x in range(targets.shape[2]):
                for y in range(targets.shape[3]):
                    if outputs[id][0][x][y] >= 0.7 and targets[id][0][x][y] == 1.0:
                        TP += 1
                    elif outputs[id][0][x][y] >= 0.7 and targets[id][0][x][y] != 1.0:
                        FP += 1
                    elif outputs[id][0][x][y] < 0.7 and targets[id][0][x][y] == 1.0:
                        FN += 1
                    elif outputs[id][0][x][y] < 0.7 and targets[id][0][x][y] != 1.0:
                        TN += 1

    #                             total += 1
    #                             if outputs[id][type][x][y] >= 0.90:
    #                                 correct += 1
    #         acc = float(correct) / total
    #         overall_total += total
    #         overall_correct += correct
    #         log_str = "\n---- [Batch %d/%d] acc = %f----\n" % (batch_i, len(dataloader), acc)
    #         print(log_str)
    #
    # # Extract labels
    # labels += targets[:, 1].tolist()
    # # Rescale target
    # targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    # targets[:, 2:] *= img_size
    #
    # imgs = Variable(imgs.type(Tensor), requires_grad=False)
    #
    # with torch.no_grad():
    #     outputs = model(imgs)
    #     outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
    #
    # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    # true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    #     overall_acc = float(overall_correct) / overall_total

    #     return overall_acc
    return TP, TN, FP, FN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/facenet.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/face.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=288, help="size of each image dimension")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_model2/face_mode_0.pth",
                        help="checkout path")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    valid_label_path = data_config["valid_label"]
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.load_state_dict(torch.load(opt.checkpoint_path))

    # Get dataloader
    resized_data_path = "data/WIDER_val/resized_images"
    resized_label_path = "data/wider_face_split/wider_face_val_bbx_gt_resized.txt"
    dataset = ListDataset(valid_path, resized_data_path, valid_label_path, resized_label_path, img_size=opt.img_size,
                          augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    print("Compute ...")
    for model_id in range(0, 1085, 5):
        # Initiate model
        model = Darknet(opt.model_def).to(device)
        if model_id <= 245:
            checkpoint_path = "checkpoint_model2/face_mode_%d.pth" % model_id
        else:
            checkpoint_path = "checkpoint_models/face_mode_%d.pth" % model_id
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

        TP, TN, FP, FN = overall_acc = evaluate(
            model,
            device,
            dataloader
        )

        acc = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("accuracy = %f" % acc)
        print("recall = %f" % recall)


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


def evaluate(model, valid_data_path, label_path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    resized_data_path = "data/WIDER_val/resized_images"
    resized_label_path = "data/wider_face_split/wider_face_val_bbx_gt_resized.txt"
    dataset = ListDataset(valid_data_path, resized_data_path, label_path, resized_label_path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.float()
        overall_total = 0
        overall_correct = 0
        with torch.no_grad():
            outputs = model(imgs)
        total = 0
        correct = 0
        for id in range(targets.shape[0]):
            for type in range(targets.shape[1]):
                for x in range(targets.shape[2]):
                    for y in range(targets.shape[3]):
                        # total += 1
                        # if targets[id][type][x][y] == 1.0 and outputs[id][type][x][y] >= 0.9:
                        #     correct += 1
                        # elif targets[id][type][x][y] == 0.0 and outputs[id][type][x][y] <= 0.2:
                        #     correct += 1
                        if targets[id][type][x][y] == 1.0:
                            total += 1
                            if outputs[id][type][x][y] >= 0.9:
                                correct += 1
        acc = float(correct) / total
        overall_total += total
        overall_correct += correct
        log_str = "\n---- [Batch %d/%d] acc = %f----\n" % (batch_i, len(dataloader), acc)
        print(log_str)
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
    overall_acc = float(overall_correct) / overall_total

    return overall_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=288, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

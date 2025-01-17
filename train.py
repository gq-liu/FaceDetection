from __future__ import division

from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/facenet.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/face.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=288, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_model2/face_mode_245.pth")
    opt = parser.parse_args()
    print(opt)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    train_label_path = data_config["train_label"]
    valid_label_path = data_config["valid_label"]
    # class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.load_state_dict(torch.load(opt.checkpoint_path))
#     model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    resized_data_path = "data/WIDER_train/Test"
    resized_label_path = "data/wider_face_split/wider_face_train_bbx_gt_resized.txt"
    dataset = ListDataset(train_path, resized_data_path, train_label_path, resized_label_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    torch.autograd.set_detect_anomaly(True)
    # metrics = [
    #     "grid_size",
    #     "loss",
    #     "x",
    #     "y",
    #     "w",
    #     "h",
    #     "conf",
    #     "cls",
    #     "cls_acc",
    #     "recall50",
    #     "recall75",
    #     "precision",
    #     "conf_obj",
    #     "conf_noobj",
    # ]
    for epoch in range(246, opt.epochs):
        model.train()
        start_time = time.time()
        loss_list = []
        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.float()
            loss, outputs = model(imgs, targets)
            loss_list.append(loss)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

#             log_str = "\n---- [Epoch %d/%d, Batch %d/%d] loss = %f----\n" % (epoch, opt.epochs, batch_i, len(dataloader)
#                                                                              , loss.item())

#             # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

#             # # Log metrics at each YOLO layer
#             # for i, metric in enumerate(metrics):
#             #     formats = {m: "%.6f" for m in metrics}
#             #     formats["grid_size"] = "%2d"
#             #     formats["cls_acc"] = "%.2f%%"
#             #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
#             #     metric_table += [[metric, *row_metrics]]
#             #
#             #     # Tensorboard logging
#             #     tensorboard_log = []
#             #     for j, yolo in enumerate(model.yolo_layers):
#             #         for name, metric in yolo.metrics.items():
#             #             if name != "grid_size":
#             #                 tensorboard_log += [(f"{name}_{j+1}", metric)]
#             #     tensorboard_log += [("loss", loss.item())]
#             #     # logger.list_of_scalars_summary(tensorboard_log, batches_done)
#             #
#             # log_str += AsciiTable(metric_table).table

#             # Determine approximate time left for epoch
#             epoch_batches_left = len(dataloader) - (batch_i + 1)
#             time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
#             log_str += f"\n---- ETA {time_left}"

#             print(log_str)

            model.seen += imgs.size(0)

        ave_loss = sum(loss_list) / len(loss_list)
        print("****** [Epoch %d/%d] average loss = %f ******" % (epoch, opt.epochs, ave_loss))

#         if epoch % opt.evaluation_interval == 0:
#             print("\n---- Evaluating Model ----")
#             # Evaluate the model on the validation set
#             overall_acc = evaluate(
#                 model,
#                 valid_data_path=valid_path,
#                 label_path=valid_label_path,
#                 iou_thres=0.5,
#                 conf_thres=0.5,
#                 nms_thres=0.5,
#                 img_size=opt.img_size,
#                 batch_size=8,
#             )
#             overall_acc_str = "\n---- [Epoch %d/%d] acc = %f----\n" % (epoch, opt.epochs, overall_acc)
#             print(overall_acc_str)
#             # logger.list_of_scalars_summary(evaluation_metrics, epoch)

#             # # Print class APs and mAP
#             # ap_table = [["Index", "Class name", "AP"]]
#             # for i, c in enumerate(ap_class):
#             #     ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
#             # print(AsciiTable(ap_table).table)
#             # print(f"---- mAP {AP.mean()}")
        #
        if epoch % opt.checkpoint_interval == 0:
            print("save_model")
            torch.save(model.state_dict(), f"checkpoint_models/face_mode_%d.pth" % epoch)

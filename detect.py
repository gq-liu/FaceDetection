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


def getAnnotation(image_origin):
    image_resized = cv2.resize(image_origin, (288, 288), interpolation=cv2.INTER_CUBIC)
    y_resize_rate = float(image_resized.shape[0]) / image_origin.shape[0]
    x_resize_rate = float(image_resized.shape[1]) / image_origin.shape[1]
    image_resized = np.expand_dims(image_resized, axis=0)
    imgs = torch.from_numpy(image_resized)
    print(imgs.shape)
    imgs = imgs.permute(0, 3, 1, 2).float()
    with torch.no_grad():
        imgs = imgs.to(device)
        output = torch.sigmoid(model(imgs))

    face_num = 0
    probability = []
    face_x_list = []
    face_y_list = []
    face_w_list = []
    face_h_list = []

    for y in range(output.shape[1]):
        for x in range(output.shape[2]):
            if output[0][0][y][x] >= 0.7:
                face_num += 1
                probability.append(output[0][0][y][x])
                face_x_list.append(((output[0][1][y][x] + x) * (288.0 / 9)) / x_resize_rate)
                face_y_list.append(((output[0][2][y][x] + y) * (288.0 / 9)) / y_resize_rate)
                face_w_list.append(output[0][3][y][x] * 288.0 / x_resize_rate)
                face_h_list.append(output[0][4][y][x] * 288.0 / y_resize_rate)
    print(output[0, 0, :, :])
    return face_num, probability, face_x_list, face_y_list, face_w_list, face_h_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/facenet.cfg", help="path to model definition file")
    parser.add_argument("--image_path", type=str, help="path to image file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=288, help="size of each image dimension")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_models/face_model_500.pth",
                        help="checkout path")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    #     data_config = parse_data_config(opt.data_config)
    # valid_path = data_config["valid"]
    # valid_label_path = data_config["valid_label"]
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    image_origin = cv2.imread(opt.image_path)
    face_num, probability, face_x_list, face_y_list, face_w_list, face_h_list = getAnnotation(image_origin)
    print(face_num)
    print(face_x_list)
    print(face_y_list)
    print(face_w_list)
    print(face_h_list)

    for face_id in range(face_num):
        x_left_up = face_x_list[face_id] - 0.5 * face_w_list[face_id] / 2
        y_left_up = face_y_list[face_id] - 0.5 * face_h_list[face_id] / 2
        x_right_bottom = x_left_up + 1 * face_w_list[face_id]
        y_right_bottom = y_left_up + 1 * face_h_list[face_id]
        image_origin = cv2.rectangle(image_origin, (x_left_up, y_left_up),
                                     (x_right_bottom, y_right_bottom), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_PLAIN
        text = "%f" % probability[face_id]
        cv2.putText(image_origin, text, (x_right_bottom, y_right_bottom), font, 1, (0, 0, 255), 1)
    cv2.imwrite("output/img.jpg", image_origin)


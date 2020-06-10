import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def resize_images(image_path, output_path, annotation_dict):
    if os.path.exists(output_path):
        return
    else:
        os.mkdir(output_path)
        dirs = os.listdir(image_path)
        for dir in dirs:
            if dir == ".DS_Store": continue
            os.mkdir(output_path + "/" + dir)
            files = os.listdir(image_path + "/" + dir)
            for file in files:
                filename = dir + "/" + file
                file_path = image_path + "/" + filename
                print(file_path)
                image_before = cv2.imread(file_path)
                image_after = cv2.resize(image_before, (288, 288), interpolation=cv2.INTER_CUBIC)
                y_resize_rate = float(image_after.shape[0]) / image_before.shape[0]
                x_resize_rate = float(image_after.shape[1]) / image_before.shape[1]
                resize_annotations(filename, y_resize_rate, x_resize_rate, annotation_dict)
                cv2.imwrite(output_path + "/" + filename, image_after)


def resize_annotations(filename, y_resize_rate, x_resize_rate, annotation_dict):
    if len(annotation_dict[filename]) == 0: return
    num_faces = annotation_dict[filename][0][0]
    for face_id in range(num_faces):
        annotation_dict[filename][face_id + 1][0] = int(annotation_dict[filename][face_id + 1][0] * x_resize_rate)
        annotation_dict[filename][face_id + 1][1] = int(annotation_dict[filename][face_id + 1][1] * y_resize_rate)
        annotation_dict[filename][face_id + 1][2] = int(annotation_dict[filename][face_id + 1][2] * x_resize_rate)
        annotation_dict[filename][face_id + 1][3] = int(annotation_dict[filename][face_id + 1][3] * y_resize_rate)
    return


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def read_annotations(label_path):
    labels = {}
    with open(label_path, "r") as file:
        lines = file.readlines()
        curr_img = ""
        for line in lines:
            line = line[:-1]
            if line.endswith("jpg"):
                labels[line] = []
                curr_img = line
            else:
                line_list = line.strip().split(" ")
                line_list = [int(val) for val in line_list]
                labels[curr_img].append(line_list)
    return labels


def write_annotations(annotations, output_path):
    with open(output_path, 'a') as file:
        for key in annotations:
            file.write(key + "\n")
            for list in annotations[key]:
                line = " ".join([str(val) for val in list])
                file.write(line + "\n")

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, label_path, img_size=288, augment=True, multiscale=True, normalized_labels=True):
        self.data_path = "data/WIDER_train/resized_images"
        if not os.path.exists(self.data_path):
            self.img_labels_dict = read_annotations(label_path)
            resize_images(list_path, self.data_path, self.img_labels_dict)
            write_annotations(self.img_labels_dict, "data/wider_face_split/wider_face_train_bbx_gt_resized.txt")

        self.img_labels_dict = read_annotations("data/wider_face_split/wider_face_train_bbx_gt_resized.txt")
        self.img_files = []
        self.label_files = []
        dirs = os.listdir(self.data_path)
        for dir in dirs:
            if dir == ".DS_Store": continue
            files = os.listdir(self.data_path + "/" + dir)
            for file in files:
                filename = dir + "/" + file
                print(filename)
                file_path = self.data_path + "/" + filename
                image = cv2.imread(filename)
                # image_resized = resize(image, img_size)
                self.img_files.append(image)
                self.label_files.append(self.img_labels_dict[filename])
        print(len(self.img_files))
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()
        print(self.label_files[0])

        # self.label_files = [
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #     for path in self.img_files
        # ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
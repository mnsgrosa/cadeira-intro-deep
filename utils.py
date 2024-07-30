import random
import cv2
import os
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torchmetrics import Accuracy

def get_random_file(path):
    file = random.choice(os.listdir(path))
    return os.path.join(path, file)

def file_to_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError("Image file not found.")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_random_image(path, picked_list):
    picked = False
    while not picked:
        random_file = get_random_file(images_path)
        if random_file not in picked_list:
            picked_list.append(random_file)
    return file_to_image(random_file)

def load_images(data_path, train=True, train_size=0.8):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    picked_list = []

    label_translator = {'negative': 0, 'positive': 1}

    negative_dir = os.path.join(data_path, 'negative')
    positive_dir = os.path.join(data_path, 'positive')

    possible_dirs = {'negative': negative_dir, 'positive': positive_dir}

    total_files = len(os.listdir(negative_dir)) + len(os.listdir(positive_dir))
    total_train_files = int(total_files * train_size)
    total_test_files = total_files - total_train_files

    # Function to collect random images and labels
    def collect_images_and_labels(num_files, images_list, labels_list):
        while len(images_list) < num_files:
            label, images_path = random.choice(list(possible_dirs.items()))
            file_path = get_random_file(images_path)
            if file_path not in picked_list:
                picked_list.append(file_path)
                images_list.append(file_to_image(file_path))
                labels_list.append(label_translator[label])

    if train:
        collect_images_and_labels(total_train_files, train_images, train_labels)
        return train_images, train_labels
    else:
        collect_images_and_labels(total_test_files, test_images, test_labels)
        return test_images, test_labels


def new_load_images(data_path = '/teamspace/studios/this_studio/brain_tumor_final/Brain_Tumor/Brain_Tumor', dataset_path = 'Brain_tumor_final/Brain_Tumor.csv', train_size = 0.8):
    df = dataset_path.read_csv(dataset_path)
    df.set_index('Image', inplace = True)
    images = os.listdir(data_path)
    total_images = len(images)
    train_size_images = total_images * train_size
    test_size_images = total_images - train_size_images
    
    train_images = []
    temp_image_train = []
    train_labels = []
    test_images = []
    temp_image_test = []
    test_labels = []

    while len(train_images) < train_size_images:
        picked_file = os.choice(images)
        if picked_file not in temp_image_train:
            temp_image_train.append(picked_file)
            train_labels.append(df.loc[picked_file])
            train_images.append(file_to_image(os.path.join(data_path, picked_file)))
    
    while len(test_images) < test_size_images:
        picked_file = os.choice(images)
        if picked_file not in temp_image_test:
            temp_image_test.append(picked_file)
            test_labels.append(df.loc[picked_file])
            test_images.append(file_to_image(os.path.join(data_path, picked_file)))

    return train_images, train_labels, test_images, test_labels
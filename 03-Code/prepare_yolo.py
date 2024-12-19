import os
from os.path import join, basename, dirname
from glob import glob

from natsort import natsorted
import yaml
import shutil
import argparse

# Set project parameters
parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument('--patch_dir', type=str, default="../04-Data/patches_1024", help='Directory of the patches')
parser.add_argument('--name_dota_dir', type=str, default="dota_yolo", help='Name of directory of the yolo dataset')
args = parser.parse_args()

PATCH_DIR = args.patch_dir
DOTA_NAME = args.name_dota_dir
DATA_DIR = dirname(PATCH_DIR)

class_to_idx = {
           "ship" : 0, 
           "storage-tank" : 1, 
           "baseball-diamond" : 2, 
           "tennis-court" : 3, 
           "basketball-court" : 4, 
           "ground-track-field" : 5, 
           "bridge" : 6, 
           "large-vehicle" : 7, 
           "small-vehicle" : 8, 
           "helicopter" : 9, 
           "swimming-pool" : 10, 
           "roundabout" : 11, 
           "soccer-ball-field" : 12, 
           "plane" : 13, 
           "harbor" : 14,
           "container-crane" : 15,
           "no_object": 16} # for empty images

# Locate images and targets 
train_targets_path = os.path.join(PATCH_DIR, 'train_targets_patches')
train_images_path = os.path.join(PATCH_DIR, 'train_images_patches')

train_images_list = glob(join(train_images_path, "**", "*.png"),recursive=True)
train_images_list = natsorted(train_images_list, key=lambda y: y.lower())
train_targets_list = glob(join(train_targets_path, "**", "*.txt"),recursive=True)
train_targets_list = natsorted(train_targets_list, key=lambda y: y.lower())

val_targets_path = os.path.join(PATCH_DIR,'val_targets_patches')
val_images_path = os.path.join(PATCH_DIR, "val_images_patches")

val_images_list = glob(join(val_images_path, "**", "*.png"),recursive=True)
val_images_list = natsorted(val_images_list, key=lambda y: y.lower())
val_targets_list = glob(join(val_targets_path, "**", "*.txt"),recursive=True)
val_targets_list = natsorted(val_targets_list, key=lambda y: y.lower())

# Functions
def extract_hbb(pxl_coordinates: list):
    '''
    Creates a dictionary of bounding box information and returns a list of dictionaries containing bounding box information

    pxl_coordinates: List of strings
    '''
    bounding_boxes=[]
    for location in pxl_coordinates: 
        parts = location.strip().split()
        bounding_boxes.append({'label': parts[-2],
                              'x1': float(parts[0]), 
                              'y1': float(parts[1]), 
                              'x2': float(parts[2]), 
                              'y2': float(parts[3]), 
                              'x3': float(parts[4]), 
                              'y3': float(parts[5]), 
                              'x4': float(parts[6]), 
                              'y4': float(parts[7]), 
                              'difficulty': int(parts[-1])})
    return bounding_boxes

def read_txt(path):
    '''
    Reads .txt files and returns a list of strings (lines of the .txt file)

    path: Path to .txt file
    '''
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines

def transform_target(original_target_dict_list, output_file, target_size = 1024):
    '''
    Transforms the bounding box coordinates to <label x_center y_center width height> yolo format

    original_target_dict_list: Dictionaries of all bounding boxes in an image
    output_file: Path where new label text file is to be saved
    '''
    calculated_parameters = []
    for bbox_dict in original_target_dict_list:
        label = class_to_idx[bbox_dict['label']]
        if bbox_dict['x2'] >= bbox_dict['x1'] and bbox_dict['y4'] >= bbox_dict['y1']:
            center_x = ((bbox_dict['x1'] + bbox_dict['x2']) / 2) / target_size
            center_y = ((bbox_dict['y1'] + bbox_dict['y4']) / 2) / target_size
            width = (bbox_dict['x2'] - bbox_dict['x1']) / target_size
            height = (bbox_dict['y4'] - bbox_dict['y1']) / target_size
            calculated_parameters.append(f"{label} {center_x} {center_y} {width} {height}")
        else:
            print("Error: Bounding box coordinates wrong.")
            print(output_file)
            break

    if calculated_parameters:
        with open(output_file, 'w') as f:
            for params in calculated_parameters:
                f.write(params + '\n')

def transform_target_for_yolo(original_target_path, output_file_path):
    '''
    Pipeline for transforming targets for yolo

    original_target_path: Path of original DOTA target (format: x1 y1 x2 y2 x3 y3 x4 y4 label difficulty)
    output_file: Path where new yolo label text file is to be saved
    '''
    lines = read_txt(original_target_path)
    extracted_coordinates = extract_hbb(lines)
    transform_target(extracted_coordinates, output_file_path)


if not os.path.exists(join(DATA_DIR, DOTA_NAME)):
    os.mkdir(join(DATA_DIR, DOTA_NAME))
    os.mkdir(join(DATA_DIR, DOTA_NAME, "train"))
    os.mkdir(join(DATA_DIR, DOTA_NAME, "val"))
    os.mkdir(join(DATA_DIR, DOTA_NAME, "test"))

# Create train image dir and copy train images
yolo_train_images_dir = join(DATA_DIR, DOTA_NAME, "train", "images")
if not os.path.exists(yolo_train_images_dir):
    os.mkdir(yolo_train_images_dir)
    print(f'Created yolo train image directory at {yolo_train_images_dir}')
for image_path in train_images_list:
    base_name = basename(image_path)
    copy_to_path = join(yolo_train_images_dir, base_name)
    shutil.copy(image_path, copy_to_path)
print(f"Copied train images from {train_images_path} to {yolo_train_images_dir}.")

# Create train target dir and transform train targets
yolo_train_target_dir = join(DATA_DIR, DOTA_NAME, "train", "labels")
if not os.path.exists(yolo_train_target_dir):
    os.mkdir(yolo_train_target_dir)
    print(f'Created yolo train target directory at {yolo_train_target_dir}')
for original_train_target_path in train_targets_list:
    base_name = basename(original_train_target_path)
    yolo_train_target_path = join(yolo_train_target_dir, base_name)
    transform_target_for_yolo(original_train_target_path, yolo_train_target_path)
print(f"Saved yolo train targets to {yolo_train_target_dir}")

# Create val image dir and copy val images
yolo_val_images_dir = join(DATA_DIR, DOTA_NAME, "val", "images")
if not os.path.exists(yolo_val_images_dir):
    os.mkdir(yolo_val_images_dir)
    print(f'Created yolo train image directory at {yolo_val_images_dir}')
for image_path in val_images_list:
    base_name = basename(image_path)
    copy_to_path = join(yolo_val_images_dir, base_name)
    shutil.copy(image_path, copy_to_path)
print(f"Copied val images from {val_images_path} to {yolo_val_images_dir}.")

# Create val target dir and transform val targets
yolo_val_target_dir = join(DATA_DIR, DOTA_NAME, "val", "labels")
if not os.path.exists(yolo_val_target_dir):
    os.mkdir(yolo_val_target_dir)
    print(f'Created yolo train target directory at {yolo_val_target_dir}')
for original_val_target_path in val_targets_list:
    base_name = basename(original_val_target_path)
    yolo_val_target_path = join(yolo_val_target_dir, base_name)
    transform_target_for_yolo(original_val_target_path, yolo_val_target_path)
print(f"Saved yolo val targets to {yolo_val_target_dir}")

# Create dataset .yaml file
data = {
    'path': os.path.abspath(join(DATA_DIR, DOTA_NAME)),
    'train': join("train", "images"),
    'val': join("val", "images"),
    'names': {
           0 : "ship", 
           1 : "storage-tank", 
           2 : "baseball-diamond", 
           3 : "tennis-court", 
           4 : "basketball-court", 
           5 : "ground-track-field", 
           6 : "bridge", 
           7 : "large-vehicle", 
           8 : "small-vehicle", 
           9 :"helicopter", 
           10 : "swimming-pool", 
           11 : "roundabout", 
           12 : "soccer-ball-field", 
           13 :"plane", 
           14 : "harbor",
           15 : "container-crane"}
    }

# Writing data to a YAML file
save_path = os.path.join(DATA_DIR, DOTA_NAME + ".yaml")
with open(save_path, 'w') as file:
    yaml.dump(data, file)

print(f"Saved DOTA yaml file to {save_path}.")
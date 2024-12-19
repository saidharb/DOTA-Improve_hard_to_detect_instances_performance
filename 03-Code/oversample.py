import os
from os.path import join, exists, split, splitext
from glob import glob
from natsort import natsorted
import shutil
import argparse

parser = argparse.ArgumentParser(description = "Prepare patches for experiment 1.")
parser.add_argument('--patch_dir', type=str, default="../04-Data/patches_1024", help='Directory original patches to be augmented.')
parser.add_argument('--osf', type=int, default = 0, help = 'Oversampling factor: Number of times that difficult images will be copied.')
args = parser.parse_args()

PATCH_DIR = args.patch_dir
osf = args.osf

train_targets_path = join(PATCH_DIR, 'train_targets_patches')
train_images_path = join(PATCH_DIR, 'train_images_patches')

val_targets_path = join(PATCH_DIR, 'val_targets_patches')
val_images_path = join(PATCH_DIR, 'val_images_patches')

train_images_list = glob(join(train_images_path, "**", "*.png"),recursive=True)
train_images_list = natsorted(train_images_list, key=lambda y: y.lower())
train_targets_list = glob(join(train_targets_path, "**", "*.txt"),recursive=True)
train_targets_list = natsorted(train_targets_list, key=lambda y: y.lower())

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

def oversample(image_list, target_list, n = 0):
    """
    n: int ... How many copies of difficult images are to be made -> Oversample factor
    """
    counter_diff = 0
    counter_extra = 0
    counter_easy = 0
    NEW_PATCH_DIR = PATCH_DIR + f"_os_{n}"

    new_image_dir = join(NEW_PATCH_DIR, "train_images_patches")
    new_target_dir = join(NEW_PATCH_DIR, "train_targets_patches")

    #new_val_image_dir = join(NEW_PATCH_DIR, "val_images_patches")
    #new_val_target_dir = join(NEW_PATCH_DIR, "val_targets_patches")
    
    if not exists(new_image_dir):
        os.makedirs(new_image_dir)
    if not exists(new_target_dir):
        os.makedirs(new_target_dir)

    # if not exists(new_val_image_dir):
    #     os.makedirs(new_val_image_dir)
    # if not exists(new_val_target_dir):
    #     os.makedirs(new_val_target_dir)

    # shutil.copytree(val_images_path, new_val_image_dir)
    # shutil.copytree(val_targets_path, new_val_target_dir)


    for i, (image, target) in enumerate(zip(image_list, target_list)):
        print(f"{i}/{len(image_list)}", end = "\r")
        diff = False
        data = read_txt(target)
        data = extract_hbb(data)
        diff = any(bbox["difficulty"] == 1 for bbox in data)
        if diff:
            _, filename_image = split(image)
            _, filename_target = split(target)
            name_image, extension_image = splitext(filename_image)
            name_target, extension_target = splitext(filename_target)
            new_image_path = join(NEW_PATCH_DIR, "train_images_patches", filename_image)
            new_target_path = join(NEW_PATCH_DIR, "train_targets_patches", filename_target)

            shutil.copyfile(image, new_image_path)
            shutil.copyfile(target, new_target_path)

            counter_diff += 1
            
            for j in range(n):
                image_copy = f"{name_image}_copy{j}{extension_image}"
                image_copy_path = join(new_image_dir, image_copy)
                target_copy = f"{name_target}_copy{j}{extension_target}"
                target_copy_path = join(new_target_dir, target_copy)

                shutil.copyfile(image, image_copy_path)
                shutil.copyfile(target, target_copy_path)
                
                counter_extra += 1

        else:
            _, filename_image = split(image)
            _, filename_target = split(target)
            name_image, extension_image = splitext(filename_image)
            name_target, extension_target = splitext(filename_target)
            new_image_path = join(NEW_PATCH_DIR, "train_images_patches", filename_image)
            new_target_path = join(NEW_PATCH_DIR, "train_targets_patches", filename_target)

            shutil.copyfile(image, new_image_path)
            shutil.copyfile(target, new_target_path)

            counter_easy += 1

    print(f"{len(image_list)}/{len(image_list)}")
    print(f"DIFF: {counter_diff}")
    print(f"EASY: {counter_easy}")
    print(f"SUM orignal dataset: {counter_diff + counter_easy}\n")
    print(f"Oversampling factor: {osf}")
    print(f"Copied difficult images: {counter_extra}")
    print(f"SUM augmented dataset: {counter_diff + counter_easy + counter_extra}\n")
    print(f"Ratio of difficult images in original dataset: {round(100 * counter_diff / (counter_diff + counter_easy), 2)}%")
    print(f"Ratio of difficult images in augmented dataset: {round(100 * (counter_diff + counter_extra) / (counter_diff + counter_easy + counter_extra), 2)}%")
    
oversample(train_images_list, train_targets_list, osf)

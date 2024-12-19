import os
from os.path import join, basename, dirname, splitext, exists
from glob import glob
import random
import time
from natsort import natsorted
import argparse
from PIL import Image
import shutil

parser = argparse.ArgumentParser(description = "Prepare patches for experiment 2.")
parser.add_argument('--patch_dir', type=str, default="../04-Data/patches_1024_os_1", help='Directory of patch directory with copies to be augmented.')
args = parser.parse_args()

PATCH_DIR = args.patch_dir

NEW_PATCH_DIR = join(dirname(PATCH_DIR), basename(PATCH_DIR) + "_c_1") # c_1 stands for all htd (hard to detect) objects are copied once with a prob of c = 1

new_train_image_dir = join(NEW_PATCH_DIR, "train_images_patches")
new_train_target_dir = join(NEW_PATCH_DIR, "train_targets_patches")
new_val_image_dir = join(NEW_PATCH_DIR, "val_images_patches")
new_val_target_dir = join(NEW_PATCH_DIR, "val_targets_patches")

if not exists(NEW_PATCH_DIR):
    os.makedirs(new_train_image_dir)
    os.makedirs(new_train_target_dir)
    os.makedirs(new_val_image_dir)
    os.makedirs(new_val_target_dir)

train_targets_path = os.path.join(PATCH_DIR, 'train_targets_patches')
train_images_path = os.path.join(PATCH_DIR, 'train_images_patches')

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

def is_overlapping(new_box, existing_box):
    return not (
        new_box['x3'] < existing_box['x1'] or
        new_box['x1'] > existing_box['x3'] or
        new_box['y3'] < existing_box['y1'] or
        new_box['y1'] > existing_box['y3']
    )

def copy_htd_instances(train_images_list, train_targets_list):

    copied_bboxes = 0
    copies = 0
    rejected_copies = 0
    start_time = time.time()
    for i, (image, target) in enumerate(zip(train_images_list, train_targets_list)):
        print(f"{i+1}/{len(train_images_list)}", end = "\r")
        # Copies end on "copyx.png/copyx.txt"
        is_copy = splitext(basename(train_images_list[i]))[0].split("_")[-1].startswith("copy")
        if is_copy:
            
            data = read_txt(target)
            data = extract_hbb(data)
            new_target_path = join(new_train_target_dir, basename(target))
            shutil.copyfile(target, new_target_path)
            new_image_path = join(new_train_image_dir, basename(image))
            shutil.copyfile(image, new_image_path)
            copies += 1
    
            for i, bbox in enumerate(data):
                if bbox["difficulty"] == 1:
                    x_min=int(bbox["x1"])
                    y_min=int(bbox["y1"])
                    x_max=int(bbox["x3"])
                    y_max=int(bbox["y3"])
            
                    modified_image  = Image.open(new_image_path)
                    cropped = modified_image.crop((x_min, y_min, x_max, y_max))
                    bbox_width, bbox_height = cropped.size
                    
                    overlaps = True
                    iteration_counter = 0
                    while overlaps:
                        new_x_min = random.randint(0, modified_image.width - bbox_width) # so that copied bbox does not go over margin
                        new_y_min = random.randint(0, modified_image.height - bbox_height)
                        new_x_max = new_x_min + bbox_width
                        new_y_max = new_y_min + bbox_height
                        new_bbox = {
                            "x1": new_x_min,
                            "x3": new_x_max,
                            "y1": new_y_min,
                            "y3": new_y_max}
                        
                        overlaps = False
                        for existing_bbox in data:
                            if is_overlapping(new_bbox, existing_bbox):
                                overlaps = True
                                break
                        iteration_counter += 1
                        if iteration_counter > 50:
                            rejected_copies += 1
                            break

                            
                    modified_image.paste(cropped, (new_x_min, new_y_min))
                    modified_image.save(new_image_path)
                    copied_bboxes += 1
                    new_bbox_dict = {
                            "x1": new_x_min,
                            "y1": new_y_min,
                            "x2": new_x_min + bbox_width,
                            "y2": new_y_min,
                            "x3": new_x_max,
                            "y3": new_y_max,
                            "x4": new_x_min,
                            "y4": new_y_min + bbox_height,
                            "label": bbox["label"],
                            "difficulty": bbox["difficulty"]}
                    new_bbox_entry = ""
                    for key, value in new_bbox_dict.items():
                        if key != "label" and key != "difficulty":
                            new_bbox_entry += str(float(value)) + " "
                        else:
                            new_bbox_entry += str(value) + " "
                
                    new_bbox_entry = new_bbox_entry.strip()
                    with open(new_target_path, 'a') as file:
                        file.write(new_bbox_entry + '\n')
        else:
            shutil.copyfile(image, join(new_train_image_dir, basename(image)))
            shutil.copyfile(target, join(new_train_target_dir, basename(target)))
    
    print(f"{len(train_images_list)}/{len(train_images_list)}")
    print(f"Copied {copied_bboxes} bounding boxes in {copies} images.")
    print(f"There were {rejected_copies} bounding boxes which could not be copied.")
    duration = round((time.time() - start_time)/60, 2)
    print(f"Duration: {duration} minutes")

shutil.copytree(val_images_path, new_val_image_dir)
shutil.copytree(val_targets_path, new_val_target_dir)

copy_htd_instances(train_images_list, train_targets_list)
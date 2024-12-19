import os
from os.path import join
from glob import glob
from natsort import natsorted
from PIL import Image, ImageOps
import argparse
from sklearn.model_selection import train_test_split

# Get command line arguments
parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument('--data_dir', type=str, default="../04-Data", help='Directory of the data files')
parser.add_argument('--test_size', type=float, default=0.6, help='Ratio of val set to be split for test set')
parser.add_argument('--tile_size', type=int, default=1024, help='Tile size of extracted patches')
parser.add_argument('--overlap', type=int, default=256, help='Overlap between extracted patches')
args = parser.parse_args()

tile_size = args.tile_size
overlap = args.overlap
if overlap >= tile_size:
    raise ValueError(f"Overlap ({overlap}) must be smaller than tile size ({tile_size}).")
test_split_arg = args.test_size
DATA_DIR = args.data_dir

# Locate all images and targets
train_images_path = os.path.join(DATA_DIR,'images_train')
train_targets_path = os.path.join(DATA_DIR,'labels_train')

train_images_list = glob(join(train_images_path, "**", "*.png"),recursive=True)
train_images_list = natsorted(train_images_list, key=lambda y: y.lower())
train_targets_list = glob(join(train_targets_path, "**", "*.txt"),recursive=True)
train_targets_list = natsorted(train_targets_list, key=lambda y: y.lower())

val_test_images_path = os.path.join(DATA_DIR,'images_val')
val_test_targets_path = os.path.join(DATA_DIR,'labels_val')

val_test_images_list = glob(join(val_test_images_path, "**", "*.png"),recursive=True)
val_test_targets_list = glob(join(val_test_targets_path, "**", "*.txt"),recursive=True)

val_test_images_list = natsorted(val_test_images_list, key=lambda y: y.lower())
val_test_targets_list = natsorted(val_test_targets_list, key=lambda y: y.lower())

val_images_list, test_images_list, val_targets_list, test_targets_list = train_test_split(
    val_test_images_list, val_test_targets_list, test_size=test_split_arg, random_state=42
)

val_images_list = natsorted(val_images_list, key=lambda y: y.lower())
val_targets_list = natsorted(val_targets_list, key=lambda y: y.lower())

test_images_list = natsorted(test_images_list, key=lambda y: y.lower())
test_targets_list = natsorted(test_targets_list, key=lambda y: y.lower())

print(f"Train images: {len(train_images_list)}")
print(f"Train targets: {len(train_targets_list)}")
print(f"Val images: {len(val_images_list)}")
print(f"Val targets: {len(val_targets_list)}")
print(f"Test images: {len(test_images_list)}")
print(f"Test targets: {len(test_targets_list)}")

assert len(train_images_list) != 0, f"Error: Data directory does not contain train image data."
assert len(train_targets_list) != 0, f"Error: Data directory does not contain train target data."
assert len(val_images_list) != 0, f"Error: Data directory does not contain val image data."
assert len(val_targets_list) != 0, f"Error: Data directory does not contain val target data."
assert len(test_images_list) != 0, f"Error: Data directory does not contain test image data."
assert len(test_targets_list) != 0, f"Error: Data directory does not contain test target data."

# Functions
def pad_to_size(image, target_size = 1024):
    '''
    Adds zero padding to images which have a smaller size than the target size.

    image: Image to be padded
    target_size: Size of patches in pixels
    '''
    width, height = image.size
    delta_width = int(target_size) - width
    delta_height = int(target_size) - height
    padding = (0, 0, delta_width, delta_height)
    padded_image = ImageOps.expand(image, padding, fill=(0, 0, 0))
    
    return padded_image

def split_image(image_path, output_folder, tile_size=1024, overlap=256):
    '''
    Splits image into quadratic patches of specified size and overlap

    image_path: Path to image to be split
    output_folder: Folder where the patches are to be saved
    tile_size: Size of the patches in pixels
    overlap: Overlap between patches in pixels
    '''

    image = Image.open(image_path)
    mode = image.mode
    # If it is a grayscale image, convert it to RGB
    if mode == "L":
        image = image.convert("RGB")
    elif mode == "RGBA":
        image = image.convert('RGB')
    width, height = image.size
    image_name = os.path.basename(image_path).split(".")[0]
    stride = tile_size - overlap
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    tile_number = 0
    tile_counter = 0
    for top in range(0, int(height), int(stride)):
        for left in range(0, int(width), int(stride)):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            save_path = join(output_folder, f'{image_name}_{tile_size}_OL-{overlap}_x-{left}_y-{top}.png')
            if not os.path.exists(save_path):
                if (right - left) == tile_size and (bottom - top) == tile_size:
                    tile = image.crop((left, top, right, bottom))
                    tile_number += 1
                    tile.save(save_path)
                else:
                    tile = image.crop((left, top, right, bottom))
                    tile = pad_to_size(tile, tile_size)
                    tile.save(save_path)
                    tile_number += 1
            else:
                tile_counter += 1

    print(f"Created {tile_number}({tile_counter} already exist) patches from {image_name}.png.\nSaved in {output_folder}.")

def split_target(target_path, image_path, output_folder, tile_size=1024, overlap=256):
    '''
    Splits bounding boxes of an image according to the patches.

    target_path: Path to target text file
    image_path: Path to corresponding image file (not patches)
    tile_size: Size of the patches in pixels
    overlap: Overlap between patches in pixels
    '''
    hbb=read_txt(target_path)
    hbb=extract_hbb(hbb)
    target_name = os.path.basename(target_path).split(".")[0]
    image_name = os.path.basename(image_path).split(".")[0]

    image = Image.open(image_path)
    width, height = image.size

    stride = tile_size - overlap
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tile_number = 0
    tile_counter = 0
    for top in range(0, height, int(stride)):
        for left in range(0, width, int(stride)):
            bounding_boxes_per_patch_list = []
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            save_path = join(output_folder, f'{target_name}_{int(tile_size)}_OL-{int(overlap)}_x-{left}_y-{top}.txt')
            if not os.path.exists(save_path):
                for i, box in enumerate(hbb):
                    bounding_box_per_patch_dic = {}
                    if box['x1'] < left and box['x2'] < left:
                        continue
                    elif box['x1'] < left and (box['x2']>left and box['x2'] < right):
                        bounding_box_per_patch_dic['x1'] = 0
                        bounding_box_per_patch_dic['x2'] = box['x2'] - left
                        bounding_box_per_patch_dic['x3'] = box['x3'] - left
                        bounding_box_per_patch_dic['x4'] = 0
                    elif box['x1'] < left and box['x2']>right:
                        bounding_box_per_patch_dic['x1'] = 0
                        bounding_box_per_patch_dic['x2'] = right - left
                        bounding_box_per_patch_dic['x3'] = right - left  
                        bounding_box_per_patch_dic['x4'] = 0
                    elif box['x1'] >= left and box['x2'] <= right:
                        bounding_box_per_patch_dic['x1'] = box['x1'] - left
                        bounding_box_per_patch_dic['x2'] = box['x2'] - left
                        bounding_box_per_patch_dic['x3'] = box['x3'] - left
                        bounding_box_per_patch_dic['x4'] = box['x4'] - left
                    elif (box['x1'] >= left and box['x1'] <= right) and box['x2'] > right:
                        bounding_box_per_patch_dic['x1'] = box['x1'] - left
                        bounding_box_per_patch_dic['x2'] = right - left
                        bounding_box_per_patch_dic['x3'] = right - left
                        bounding_box_per_patch_dic['x4'] = box['x4'] - left
    
                    if box['y1'] < top and box['y3'] < top:
                        continue
                    elif box['y1'] < top and (box['y3'] > top and box['y3'] < bottom):
                        bounding_box_per_patch_dic['y1'] = 0
                        bounding_box_per_patch_dic['y2'] = 0
                        bounding_box_per_patch_dic['y3'] = box['y3'] - top
                        bounding_box_per_patch_dic['y4'] = box['y4'] - top
                    elif box['y1'] < top and box['y3'] > bottom:
                        bounding_box_per_patch_dic['y1'] = 0
                        bounding_box_per_patch_dic['y2'] = 0
                        bounding_box_per_patch_dic['y3'] = bottom - top
                        bounding_box_per_patch_dic['y4'] = bottom - top
                    elif box['y1'] >= top and box['y3'] <= bottom:
                        bounding_box_per_patch_dic['y3'] = box['y3'] - top
                        bounding_box_per_patch_dic['y4'] = box['y4'] - top
                        bounding_box_per_patch_dic['y1'] = box['y1'] - top
                        bounding_box_per_patch_dic['y2'] = box['y2'] - top
                    elif (box['y1'] >= top and box['y1'] <= bottom) and box['y3'] > bottom:
                        bounding_box_per_patch_dic['y1'] = box['y1'] - top
                        bounding_box_per_patch_dic['y2'] = box['y2'] - top
                        bounding_box_per_patch_dic['y3'] = bottom - top
                        bounding_box_per_patch_dic['y4'] = bottom - top
            
                    bounding_box_per_patch_dic['label'] = box['label']
                    bounding_box_per_patch_dic['difficulty'] = box['difficulty']
                    bounding_boxes_per_patch_list.append(bounding_box_per_patch_dic)
                        
                write_txt(bounding_boxes_per_patch_list, save_path)
                tile_number += 1
            else:
                tile_counter += 1
    print(f"Created {tile_number} targets ({tile_counter} already exist) for image {image_name}.\nSaved to {output_folder}.\n")

def write_txt(bbox_per_patch_list, save_path):
    '''
    Creates a .txt file with bounding box information for an image patch.

    bbox_per_patch_list: List of dictionaries containing the bounding boxes
    save_path: Path where .txt is to be saved
    '''
    with open(save_path, 'w') as file:
        for bbox in bbox_per_patch_list:
            if len(bbox.keys()) == 10:
                line_to_write = "{} {} {} {} {} {} {} {} {} {}\n".format(bbox["x1"],
                                                                      bbox["y1"],
                                                                      bbox["x2"],
                                                                      bbox["y2"],
                                                                      bbox["x3"],
                                                                      bbox["y3"],
                                                                      bbox["x4"],
                                                                      bbox["y4"],
                                                                      bbox["label"],
                                                                      bbox["difficulty"])
                file.write(line_to_write)

def extract_hbb(pxl_coordinates: list):
    '''
    Creates a dictionary of bounding box information and returns a list of dictionaries containing bounding box information

    pxl_coordinates: List of strings
    '''
    bounding_boxes=[]
    for location in pxl_coordinates[2:]: 
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

patch_dir_name = f"patches_{tile_size}"
# Split train images and targets
for image_path, target_path in zip(train_images_list, train_targets_list):
    output_folder = join(DATA_DIR, patch_dir_name, "train_images_patches", os.path.basename(image_path).split(".")[0])
    split_image(image_path, output_folder, tile_size = tile_size, overlap = overlap)
    output_folder = join(DATA_DIR, patch_dir_name, "train_targets_patches", os.path.basename(target_path).split(".")[0])
    split_target(target_path, image_path, output_folder, tile_size = tile_size, overlap = overlap)

# Split val images and targets
for image_path, target_path in zip(val_images_list, val_targets_list):
    output_folder = join(DATA_DIR, patch_dir_name, "val_images_patches", os.path.basename(image_path).split(".")[0])
    split_image(image_path, output_folder, tile_size = tile_size, overlap = overlap)
    output_folder = join(DATA_DIR, patch_dir_name, "val_targets_patches", os.path.basename(target_path).split(".")[0])
    split_target(target_path, image_path, output_folder, tile_size = tile_size, overlap = overlap)

# Split test images and targets
for image_path, target_path in zip(test_images_list, test_targets_list):
    output_folder = join(DATA_DIR, patch_dir_name, "test_images_patches", os.path.basename(image_path).split(".")[0])
    split_image(image_path, output_folder, tile_size = tile_size, overlap = 0)
    output_folder = join(DATA_DIR, patch_dir_name, "test_targets_patches", os.path.basename(target_path).split(".")[0])
    split_target(target_path, image_path, output_folder, tile_size = tile_size, overlap = 0)
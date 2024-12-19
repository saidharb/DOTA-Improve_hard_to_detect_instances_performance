import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np


class ToTensor(object):
    def __call__(self, image):
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float()/255.0        
        
        return image
    
class DotaDataset(Dataset):
    def __init__(self, images_path, targets_path, transform = ToTensor()):
        
        self.images_path = images_path
        self.targets_path = targets_path
        self.transform = transform

        self.class_to_idx = {
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
               "no_object": 16}

        self.colors = {
            'container-crane': 'blue',
            'ship': 'red',
            'storage-tank': 'green',
            'baseball-diamond': 'yellow',
            'tennis-court': 'purple',
            'basketball-court': 'orange',
            'ground-track-field': 'pink',
            'bridge': 'turquoise',
            'large-vehicle': 'brown',
            'small-vehicle': 'azure',
            'helicopter': 'lime',
            'swimming-pool': 'cyan',
            'roundabout': 'magenta',
            'soccer-ball-field': 'gold',
            'plane': 'lime',
            'harbor': 'indigo',
            'no_object': 'black'}

    def __len__(self):
        return(len(self.images_path))

    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx])
        target = self.targets_path[idx]
        if self.transform:
            image = self.transform(image)

        hbb=self.read_txt(self.targets_path[idx])
        hbb=self.extract_hbb(hbb)

        boxes = []
        labels = []

        # What happens if boxes and labels are empty lists? Does this work during training?
        for box in hbb:
            xmin, xmax, ymin, ymax, label = box['x_min'], box['x_max'], box['y_min'], box['y_max'], box['label']
            boxes.append([xmin, ymin, xmax, ymax])
            id = self.class_to_idx[label]
            labels.append(id)

        if len(boxes) == 0:
            boxes.append([0, 0, 0, 0])  # Dummy box
            labels.append(-1)            # Dummy label
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        targets = {
        "boxes": boxes,
        "labels": labels}
        
        return image, targets

    def read_txt(self, path):
        '''
        Reads a text file
        Returns a list of lines in the text file
        path: Path to text file
        '''
        with open(path, 'r') as file:
            lines = file.readlines()
        return lines

    def get_image_path(self, idx):
        '''
        Returns image path at the index
        idx: Index of image path to return
        '''
        return self.images_path[idx]

    def get_unique_labels(self, target_path):
        '''
        Returns the unique labels of targets in an image
        taget_path: Path to the target
        '''
        unique_names = set()
        txt_file=self.read_txt(target_path)
        hbb_list=self.extract_hbb(txt_file)
        for d in hbb_list:
            name = d["label"]
            unique_names.add(name)
        unique_names=list(unique_names)
        return unique_names

    def create_custom_legend(self, labels,axes):
        '''
        Creates a legend for the objects in an image
        labels: Labels of the objects
        '''
        bbox_to_anchor=(1.04, 1)
        handles = []
        color_list=[]
        for label in labels:
            color_list.append(self.colors[label])
        for color, label in zip(color_list, labels):
            handle = Rectangle((0, 0), 1, 1, color=color, label=label)
            handles.append(handle)
        axes.legend(handles=handles,  loc="upper left", bbox_to_anchor=bbox_to_anchor)

    def extract_hbb(self, pxl_coordinates: list):
        '''
        Returns a dictionary of bounding box coordinates from a text file
        pxl_coordinates: List of bounding box strings
        '''
        bounding_boxes=[]
        for location in pxl_coordinates:
            parts = location.strip().split()
            x_min=float(parts[0])
            y_min=float(parts[1])
            x_max=float(parts[4])
            y_max=float(parts[5])
            class_label=parts[8]
            bounding_boxes.append({'label':class_label,
                                  'x_min':x_min,
                                  'y_min':y_min,
                                  'x_max':x_max,
                                  'y_max':y_max})
        return bounding_boxes

    def get_number_of_instances(self, target_path):
        '''
        Returns the number of objects in an image.
        target_path: Path to the target of the image
        '''
        unique_names=self.get_unique_labels(target_path)
        txt_file=self.read_txt(target_path)
        hbb_list=self.extract_hbb(txt_file)
        dic={item: 0  for item in unique_names}
        for box in hbb_list:
            if box["label"] in dic.keys():
                dic[box["label"]]+=1
        return dic

    def visualize(self, idx):
        '''
        Shows the image at the index and the objects 
        idx: Index of image to display
        '''
        print("Image path: " + self.images_path[idx])
        print("Target path: " + self.targets_path[idx])
    
        image = Image.open(self.images_path[idx])
        #if self.transform:
        #    image = self.transform(image)
        #image = image.permute(1, 2, 0).numpy()  # Change order to HxWxC
    
        fig, ax = plt.subplots()
        ax.imshow(image)
        hbb=self.read_txt(self.targets_path[idx])
        hbb=self.extract_hbb(hbb)
        for box in hbb:
            if box['label'] in self.colors:
                color=self.colors[box['label']]
            xmin, xmax, ymin, ymax, label = box['x_min'], box['x_max'], box['y_min'], box['y_max'], box['label']
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        self.create_custom_legend(self.get_unique_labels(self.targets_path[idx]),ax)
        plt.show()
        example=self.get_number_of_instances(self.targets_path[idx])
        for key, value in example.items():
            print(f"{key}: {value}")
        print("\n")
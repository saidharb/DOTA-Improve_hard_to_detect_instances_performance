import os
from os.path import join
from glob import glob
import time

import numpy as np
from natsort import natsorted
import torch
from ultralytics import YOLO
import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(description = "Evaluate model.")
parser.add_argument('--patch_dir', type=str, default="../04-Data/patches_1024", help='Directory of the extracted patches')
parser.add_argument('--model_path', type=str, default="../best.pt", help='Path to model')
parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--max_img', type=int, default=-1, help='Maximum number of images to process')
parser.add_argument('--save_interval', type=int, default=100, help='Number of intervals after which data is saved')
parser.add_argument('--result_dir', type=str, default="Results", help='Directory to save results in')
args = parser.parse_args()

PATCH_DIR = args.patch_dir
RESULT_DIR = args.result_dir
model_path_arg = args.model_path
iou_arg = args.iou
conf_arg = args.conf
max_img_arg = args.max_img
save_interval_arg = args.save_interval

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
           "container-crane" : 15}
idx_to_class = {
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

test_targets_path = os.path.join(PATCH_DIR,'test_targets_patches')
test_images_path = os.path.join(PATCH_DIR, "test_images_patches")

test_images_list = glob(join(test_images_path, "**", "*.png"),recursive=True)
test_images_list = natsorted(test_images_list, key=lambda y: y.lower())
test_targets_list = glob(join(test_targets_path, "**", "*.txt"),recursive=True)
test_targets_list = natsorted(test_targets_list, key=lambda y: y.lower())

print(f"Test images: {len(test_images_list)}")
print(f"Test targets: {len(test_targets_list)}")

def extract_hbb(pxl_coordinates: list):
    '''
    Creates a dictionary of bounding box information and returns a list of dictionaries containing bounding box information

    pxl_coordinates: List of strings
    '''
    bounding_boxes=[]
    for location in pxl_coordinates: 
        parts = location.strip().split()
        bounding_boxes.append({'label': parts[-2],
                              'x1': float(parts[0]), # x_min
                              'y1': float(parts[1]), # y_min
                              'x2': float(parts[2]), 
                              'y2': float(parts[3]), 
                              'x3': float(parts[4]), # x_max
                              'y3': float(parts[5]), # y_max
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

def calculate_iou(box1, box2):
    """
    Calculates the IoU between two bounding boxes.
    """
    x_min_inter = torch.max(box1[0], box2[0])
    y_min_inter = torch.max(box1[1], box2[1])
    x_max_inter = torch.min(box1[2], box2[2])
    y_max_inter = torch.min(box1[3], box2[3])
    
    inter_area = (x_max_inter - x_min_inter).clamp(0) * (y_max_inter - y_min_inter).clamp(0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0  
        
    iou = inter_area / union_area
    return iou

def write_to_csv_pr_curve(prec_recall_per_class_threshold, iou_threshold, name = "precision_recall_curve"):
    '''
    Creates csv data for precision-recall curves.
    '''
    save_dir = join(RESULT_DIR, "Precision-Recall-Curve")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, name + ".csv")
    fieldnames = ['class','iou threshold', 'confidence threshold', 'precision', 'recall']
    
    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cls, threshold_dict in prec_recall_per_class_threshold.items():
            for threshold, metrics in threshold_dict.items():
                row = {'class': cls, 
                       'iou threshold': iou_threshold, 
                       'confidence threshold': threshold,
                        **metrics}
                writer.writerow(row)
            
    print(f'Data saved to {file_name}')

def write_to_csv_pr(prec_recall_per_class, iou_threshold, confidence_threshold, name = "precision_recall"):
    '''
    Creates csv data for precision and recall metrics.
    '''
    save_dir = join(RESULT_DIR, "Precision-Recall")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, name + ".csv")
    fieldnames = ['class','iou threshold', 'confidence threshold', 'precision', 'recall', 'f1']
    
    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cls, metrics in prec_recall_per_class.items():
            row = {'class': cls, 
                   'iou threshold': iou_threshold, 
                   'confidence threshold': confidence_threshold,  
                   **metrics}
            writer.writerow(row)
            
    print(f'Data saved to {file_name}')

def classify_predictions(targets, predictions, iou_threshold=0.5, confidence_threshold=0.5):
    '''
    Classifies the predictions of a given prediction according to the target. 
    Returns TP, FP and FN of prediction per difficulty.
    '''
    matched_predictions = set()
    classes = [
        "ship", "storage-tank", "baseball-diamond", "tennis-court", 
        "basketball-court", "ground-track-field", "bridge", 
        "large-vehicle", "small-vehicle", "helicopter", "swimming-pool", 
        "roundabout", "soccer-ball-field", "plane", "harbor", 
        "container-crane"
    ]
    metrics_per_class = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in classes}
    metrics_per_class_diff = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in classes}
    metrics_per_class_easy = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in classes}

    # CASE 1: No targets -> All predictions are FP
    # Consideration: If there are no targets, we dont know if difficult or not 
    if targets.numel() == 0:
        for pred in predictions:
            if pred[4] >= confidence_threshold:
                predicted_label = idx_to_class[int(pred[5])]
                metrics_per_class[predicted_label]["FP"] += 1
                metrics_per_class_diff[predicted_label]["FP"] += 1
                metrics_per_class_easy[predicted_label]["FP"] += 1
        return metrics_per_class, metrics_per_class_diff, metrics_per_class_easy
    
    # CASE 2: No predictions -> All targets are FN
    if predictions.numel() == 0:
        for target in targets:
            difficult = False
            if target[6] == 1:
                difficult = True
            true_label = idx_to_class[int(target[5])]
            metrics_per_class[true_label]["FN"] += 1
            if difficult:
                metrics_per_class_diff[true_label]["FN"] += 1
            else:
                metrics_per_class_easy[true_label]["FN"] += 1
        return metrics_per_class, metrics_per_class_diff, metrics_per_class_easy

    # CASE 3: Matching predictions -> Either TP or FN
    valid_predictions = [pred for pred in predictions if pred[4] >= confidence_threshold]
    valid_predictions.sort(key=lambda x: x[4], reverse=True)

    for target in targets:
        best_iou = 0
        best_j = -1
        true_label = target[5]
        difficult = False
        if target[6] == 1:
            difficult = True
        for j, pred in enumerate(valid_predictions):
            iou = calculate_iou(target, pred)
            pred_label = pred[5]
            if pred_label == true_label and iou > best_iou:
                best_iou = iou
                best_j = j
                
        if best_j != -1 and best_iou >= iou_threshold:
            predicted_label = valid_predictions[best_j].tolist()[5]
            if best_j not in matched_predictions: # So that one prediction can only predict one instance
                predicted_label = idx_to_class[predicted_label]
                metrics_per_class[predicted_label]["TP"] += 1
                matched_predictions.add(best_j)
                if difficult:
                    metrics_per_class_diff[predicted_label]["TP"] += 1
                else:
                    metrics_per_class_easy[predicted_label]["TP"] += 1
        else:
            true_label = idx_to_class[true_label.item()]
            metrics_per_class[true_label]["FN"] += 1
            if difficult:
                metrics_per_class_diff[true_label]["FN"] += 1
            else:
                metrics_per_class_easy[true_label]["FN"] += 1

    # CASE 4: Not matching predictions -> Duplicates Rest is FP
    for j, pred in enumerate(valid_predictions):
        predicted_label = pred.tolist()[5]
        predicted_label = idx_to_class[predicted_label]
        if j not in matched_predictions:
            metrics_per_class[predicted_label]["FP"] += 1
            if difficult:
                metrics_per_class_diff[predicted_label]["FP"] += 1
            else:
                metrics_per_class_easy[predicted_label]["FP"] += 1

    return metrics_per_class, metrics_per_class_diff, metrics_per_class_easy

class ObjectDetectionRunningScore():
    '''
    Class to update and get metrics.
    '''
    def __init__(self):
        self.classes = [
        "ship", "storage-tank", "baseball-diamond", "tennis-court", 
        "basketball-court", "ground-track-field", "bridge", 
        "large-vehicle", "small-vehicle", "helicopter", "swimming-pool", 
        "roundabout", "soccer-ball-field", "plane", "harbor", 
        "container-crane"]
        self.confidence_thresholds = np.arange(0.0, 1.01, 0.01)
        self.confidence_thresholds = self.confidence_thresholds[::-1]
        self.metrics_per_class = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in self.classes}
        self.pr_metrics_per_class = {
            cls: {round(threshold, 2): {"TP": 0, "FP": 0, "FN": 0} for threshold in self.confidence_thresholds}
            for cls in self.classes}

    def update(self, update_dict):
        for cls, metric_dict in update_dict.items():
            self.metrics_per_class[cls]["TP"] += metric_dict.get("TP", 0)
            self.metrics_per_class[cls]["FP"] += metric_dict.get("FP", 0)
            self.metrics_per_class[cls]["FN"] += metric_dict.get("FN", 0)
    
    def get_metrics(self):
        metrics = {}
        for cls, metric_dict in self.metrics_per_class.items():
            tp = metric_dict["TP"]
            fp = metric_dict["FP"]
            fn = metric_dict["FN"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
        return metrics

    def update_pr_curve(self, update_dict):
        for cls, threshold_dict in self.pr_metrics_per_class.items():
            for threshold, metric_dict in threshold_dict.items():
                self.pr_metrics_per_class[cls][threshold]["TP"] += update_dict[cls][threshold]["TP"]
                self.pr_metrics_per_class[cls][threshold]["FP"] += update_dict[cls][threshold]["FP"]
                self.pr_metrics_per_class[cls][threshold]["FN"] += update_dict[cls][threshold]["FN"]

    def get_pr_curves(self):
        precision_recall_dict = {
            cls: {round(threshold, 2): {"precision": [], "recall": []} for threshold in self.confidence_thresholds} 
            for cls in self.classes
        }
        for cls, threshold_dict in self.pr_metrics_per_class.items():
            for threshold, metric_dict in threshold_dict.items():
                tp = metric_dict["TP"]
                fp = metric_dict["FP"]
                fn = metric_dict["FN"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precision_recall_dict[cls][threshold]["precision"] = precision
                precision_recall_dict[cls][threshold]["recall"] = recall
        return precision_recall_dict

    def compute_class_metrics(self):
        class_metrics = {cls: {"Total": 0, "TP": 0, "FP": 0, "FN": 0} for cls in self.classes}
        for cls, threshold_dict in self.pr_metrics_per_class.items():
            for threshold, metric_dict in threshold_dict.items():
                class_metrics[cls]["TP"] += metric_dict["TP"]
                class_metrics[cls]["FP"] += metric_dict["FP"]
                class_metrics[cls]["FN"] += metric_dict["FN"]
                class_metrics[cls]["Total"] += metric_dict["TP"] + metric_dict["FP"] + metric_dict["FN"]
        return class_metrics
    
def target_to_yolo_val_tensor(path):
    '''
    Transforms a target txt file to the same format as yolo predictions for easier comparison.
    '''
    lines = read_txt(path)
    bb_list = extract_hbb(lines)
    if len(bb_list) != 0:
        data = torch.zeros(len(bb_list), 7)
        for i, bb_dict in enumerate(bb_list):
            x_min = bb_dict["x1"] 
            y_min = bb_dict["y1"] 
            x_max = bb_dict["x3"]  
            y_max = bb_dict["y3"] 
            label = bb_dict["label"]
            difficulty = bb_dict["difficulty"]
            conf = 1.0
            bb_data = torch.tensor([x_min, y_min, x_max, y_max, conf, class_to_idx[label], difficulty])
            data[i] = bb_data
    else:
        return torch.tensor([])
    return data

def csv_to_dict(csv_file_path):
    '''
    Creates a dictionary from a csv file.
    '''
    df = pd.read_csv(csv_file_path)
    result_dict = {}
    for obj_class in df['class'].unique():
        class_df = df[df['class'] == obj_class]
        class_dict = {}
        for _, row in class_df.iterrows():
            confidence = row['confidence threshold']
            precision = row['precision']
            recall = row['recall']
            class_dict[confidence] = {'precision': precision, 'recall': recall}
        result_dict[obj_class] = class_dict
    return result_dict

def calculate_precision_recall(
    model, 
    iou_threshold = 0.5, 
    conf_threshold = 0.5, 
    max_images = -1,
    save_interval = 100):
    '''
    Calculates precision and recall per class and difficulty.
    '''
    start_time = time.time()
    all_metrics = ObjectDetectionRunningScore()
    easy_metrics = ObjectDetectionRunningScore()
    hard_metrics = ObjectDetectionRunningScore()
    
    print("Calculating metrics per class:")
    for i, paths in enumerate(zip(test_images_list, test_targets_list)):
        if i+1 != len(test_targets_list):
            print(f"{i+1}/{len(test_targets_list)}", end = "\r")
            
        image_path = paths[0]
        target_path = paths[1]
        
        prediction = model([image_path], verbose = False)
        yolo_prediction = prediction[0].boxes.data
        yolo_target = target_to_yolo_val_tensor(target_path)

        dict, dict_diff, dict_easy = classify_predictions(
            yolo_target, 
            yolo_prediction, 
            iou_threshold = iou_threshold, 
            confidence_threshold = conf_threshold)
        
        all_metrics.update(dict)
        easy_metrics.update(dict_easy)
        hard_metrics.update(dict_diff)
        if i%save_interval == 0 and i != 0:
            print(f"{i+1}/{len(test_images_list)}")
            all_pr = all_metrics.get_metrics()
            easy_pr = easy_metrics.get_metrics()
            hard_pr = hard_metrics.get_metrics()
            write_to_csv_pr(all_pr, iou_threshold, conf_threshold, "all_pr")
            write_to_csv_pr(easy_pr, iou_threshold, conf_threshold, "easy_pr")
            write_to_csv_pr(hard_pr, iou_threshold, conf_threshold, "hard_pr")
        if i+1 == max_images:
            break

    print(f"{len(test_targets_list)}/{len(test_images_list)}")
    all_pr = all_metrics.get_metrics()
    easy_pr = easy_metrics.get_metrics()
    hard_pr = hard_metrics.get_metrics()

    write_to_csv_pr(all_pr, iou_threshold, conf_threshold, "all_pr")
    write_to_csv_pr(easy_pr, iou_threshold, conf_threshold, "easy_pr")
    write_to_csv_pr(hard_pr, iou_threshold, conf_threshold, "hard_pr")
    
    duration = time.time() - start_time
    print(f"Duration: {round(duration/60)} minutes")
    print("DONE\n")
   
    return all_pr, easy_pr, hard_pr

def count_numbers_with_dict(numbers):
    '''
    Calculates the number of instances in a dict.
    '''
    counts = {}
    
    for number in numbers:
        if number in counts:
            counts[number] += 1
        else:
            counts[number] = 1
            
    for key, value in counts.items():
        print(idx_to_class[key], value, sep =  ": ")

def filter_and_sum_non_zero_classes(data):
    """
    Filter out classes with non-zero entries in the dictionary and compute the sum of TP, FP, and FN.
    """
    non_zero_classes = {}

    for class_name, metrics in data.items():
        total_sum = sum(metrics.values())
        
        if total_sum > 0:
            non_zero_classes[class_name] = metrics
            non_zero_classes[class_name]['Total'] = total_sum

    for class_name, metrics in non_zero_classes.items():
        tp, fp, fn = metrics['TP'], metrics['FP'], metrics['FN']
        total = tp + fp + fn
        print(f"{class_name}: TP={tp}, FP={fp}, FN={fn}, Total={total}")

def calculate_precision_recall_curves(
    model, 
    test_images_list, 
    test_targets_list,
    iou_threshold = 0.5,
    save_interval = 100,
    sanity_check = [False, False, False], # All, diff, easy
    classes_sanity_check = [], 
    max_images = -1):
    '''
    Calculates precision and recall for every class per difficulty and confidence threshold.
    '''
    
    classes = [
        "ship", "storage-tank", "baseball-diamond", "tennis-court", 
        "basketball-court", "ground-track-field", "bridge", 
        "large-vehicle", "small-vehicle", "helicopter", "swimming-pool", 
        "roundabout", "soccer-ball-field", "plane", "harbor", 
        "container-crane"
    ]
    scores_all = ObjectDetectionRunningScore()
    scores_diff = ObjectDetectionRunningScore()
    scores_easy = ObjectDetectionRunningScore()
    counter_dict = {}
    confidence_thresholds = np.arange(0.0, 1.01, 0.01)
    confidence_thresholds = confidence_thresholds[::-1]
    metric_dict_all = {
        cls: {round(threshold, 2): {"TP": 0, "FP": 0, "FN": 0} for threshold in confidence_thresholds}
        for cls in classes
    }
    metric_dict_diff = {
        cls: {round(threshold, 2): {"TP": 0, "FP": 0, "FN": 0} for threshold in confidence_thresholds}
        for cls in classes
    }
    metric_dict_easy = {
        cls: {round(threshold, 2): {"TP": 0, "FP": 0, "FN": 0} for threshold in confidence_thresholds} 
        for cls in classes
    }
    start_time = time.time()
    print("Calculating Precision-Recall Curve per class:\n")
    for i, (image_path, target_path) in enumerate(zip(test_images_list, test_targets_list)):
        print(f"{i+1}/{len(test_images_list)}", end = "\r")
        
        prediction = model([image_path], verbose = False)
        yolo_prediction = prediction[0].boxes.data    
        yolo_target = target_to_yolo_val_tensor(target_path)
        
        if any(sanity_check):
            print(f"\nTarget path: {target_path}")
        if sanity_check[0]:
            print("TARGET for ALL")
            count_numbers_with_dict(yolo_target[:,5].tolist())
            print("PREDICTION for ALL")
            count_numbers_with_dict(prediction[0].boxes.cls.tolist())
        if sanity_check[1]:
            print("TARGET for HARD")
            mask = yolo_target[:, 6] == 1
            diff_yolo_target = yolo_target[mask]
            count_numbers_with_dict(diff_yolo_target[:,5].tolist())
            print("PREDICTION for ALL")
            count_numbers_with_dict(prediction[0].boxes.cls.tolist())
        if sanity_check[2]:
            print("TARGET for EASY")
            mask = yolo_target[:, 6] == 0
            easy_yolo_target = yolo_target[mask]
            count_numbers_with_dict(easy_yolo_target[:,5].tolist())
            print("PREDICTION for ALL")
            count_numbers_with_dict(prediction[0].boxes.cls.tolist())
        
        for threshold in confidence_thresholds:
            threshold = round(threshold, 2)
            dict_all, dict_diff, dict_easy = classify_predictions(yolo_target, 
                                                              yolo_prediction, 
                                                              iou_threshold = iou_threshold, 
                                                              confidence_threshold = threshold)
            
            if any(sanity_check):
                print(f"\nConfidence threshold = {threshold}")   
            if sanity_check[0]:
                print("ALL")
                filter_and_sum_non_zero_classes(dict_all)
            if sanity_check[1]:
                print("HARD")
                filter_and_sum_non_zero_classes(dict_diff)
            if sanity_check[2]:
                print("EASY")
                filter_and_sum_non_zero_classes(dict_easy)
                
            for cls, dict in dict_all.items():
                metric_dict_all[cls][threshold]["TP"] += dict.get('TP', 0) 
                metric_dict_all[cls][threshold]["FP"] += dict.get('FP', 0) 
                metric_dict_all[cls][threshold]["FN"] += dict.get('FN', 0)
                
                if sanity_check[0]:
                    if cls in classes_sanity_check:
                        print(f"Metrics for ALL class '{cls}':")
                        if cls not in counter_dict.keys():
                            counter_dict[cls] = {}
                        if "all" not in counter_dict[cls].keys():
                            counter_dict[cls]["all"] = {"counter_tp": 0, "counter_fp": 0, "counter_fn": 0}
                        counter_dict[cls]["all"]["counter_tp"] += dict["TP"]
                        counter_dict[cls]["all"]["counter_fp"] += dict["FP"]
                        counter_dict[cls]["all"]["counter_fn"] += dict["FN"]
                        
                        print(f"SUM -> TP: {counter_dict[cls]["all"]["counter_tp"]} FP: {counter_dict[cls]["all"]["counter_fp"]} FN:{counter_dict[cls]["all"]["counter_fn"]}")
                        print(f"UPDATE -> TP:{metric_dict_all[cls][threshold]["TP"]} FP: {metric_dict_all[cls][threshold]["FP"]} FN: {metric_dict_all[cls][threshold]["FN"]}")
                        print(f"Recall: {dict["TP"]/(dict["TP"]+dict["FN"]) if (dict["TP"]+dict["FN"]) > 0 else 0.0}")
                        print(f"Precision: {dict["TP"]/(dict["TP"]+dict["FP"]) if (dict["TP"]+dict["FP"]) > 0 else 1.0}")
               
            for cls, dict in dict_diff.items():
                metric_dict_diff[cls][threshold]["TP"] += dict.get('TP', 0) 
                metric_dict_diff[cls][threshold]["FP"] += dict.get('FP', 0) 
                metric_dict_diff[cls][threshold]["FN"] += dict.get('FN', 0)
                
                if sanity_check[1]:
                    if cls in classes_sanity_check:
                        print(f"Metrics for HARD class '{cls}':")
                        if cls not in counter_dict.keys():
                            counter_dict[cls] = {}
                        if "hard" not in counter_dict[cls].keys():
                            counter_dict[cls]["hard"] = {"counter_tp": 0, "counter_fp": 0, "counter_fn": 0}
                        counter_dict[cls]["hard"]["counter_tp"] += dict["TP"]
                        counter_dict[cls]["hard"]["counter_fp"] += dict["FP"]
                        counter_dict[cls]["hard"]["counter_fn"] += dict["FN"]
                        
                        print(f"SUM -> TP: {counter_dict[cls]["hard"]["counter_tp"]} FP: {counter_dict[cls]["hard"]["counter_fp"]} FN:{counter_dict[cls]["hard"]["counter_fn"]}")
                        print(f"UPDATE -> TP:{metric_dict_diff[cls][threshold]["TP"]} FP: {metric_dict_diff[cls][threshold]["FP"]} FN: {metric_dict_diff[cls][threshold]["FN"]}")
                        print(f"Recall: {dict["TP"]/(dict["TP"]+dict["FN"]) if (dict["TP"]+dict["FN"]) > 0 else 0.0}")
                        print(f"Precision: {dict["TP"]/(dict["TP"]+dict["FP"]) if (dict["TP"]+dict["FP"]) > 0 else 1.0}")
                
            for cls, dict in dict_easy.items():
                metric_dict_easy[cls][threshold]["TP"] += dict.get('TP', 0) 
                metric_dict_easy[cls][threshold]["FP"] += dict.get('FP', 0) 
                metric_dict_easy[cls][threshold]["FN"] += dict.get('FN', 0)
                
                if sanity_check[2]:
                    if cls in classes_sanity_check:
                        print(f"Metrics for EASY class '{cls}':")
                        if cls not in counter_dict.keys():
                            counter_dict[cls] = {}
                        if "easy" not in counter_dict[cls].keys():
                            counter_dict[cls]["easy"] = {"counter_tp": 0, "counter_fp": 0, "counter_fn": 0}
                        counter_dict[cls]["easy"]["counter_tp"] += dict["TP"]
                        counter_dict[cls]["easy"]["counter_fp"] += dict["FP"]
                        counter_dict[cls]["easy"]["counter_fn"] += dict["FN"]
                        
                        print(f"SUM -> TP: {counter_dict[cls]["easy"]["counter_tp"]} FP: {counter_dict[cls]["easy"]["counter_fp"]} FN:{counter_dict[cls]["easy"]["counter_fn"]}")
                        print(f"UPDATE -> TP:{metric_dict_easy[cls][threshold]["TP"]} FP: {metric_dict_easy[cls][threshold]["FP"]} FN: {metric_dict_easy[cls][threshold]["FN"]}")
                        print(f"Recall: {dict["TP"]/(dict["TP"]+dict["FN"]) if (dict["TP"]+dict["FN"]) > 0 else 0.0}")
                        print(f"Precision: {dict["TP"]/(dict["TP"]+dict["FP"]) if (dict["TP"]+dict["FP"]) > 0 else 1.0}")
        
        scores_all.update_pr_curve(metric_dict_all)
        scores_diff.update_pr_curve(metric_dict_diff)
        scores_easy.update_pr_curve(metric_dict_easy)
        if i%save_interval == 0 and i != 0:
            print(f"{i+1}/{len(test_images_list)}")
            sc_all = scores_all.get_pr_curves()
            sc_diff = scores_diff.get_pr_curves()
            sc_easy = scores_easy.get_pr_curves()
        
            write_to_csv_pr_curve(sc_all, iou_threshold, name = "all_pr_curve")
            write_to_csv_pr_curve(sc_diff, iou_threshold, name = "diff_pr_curve")
            write_to_csv_pr_curve(sc_easy, iou_threshold, name = "easy_pr_curve")
        if i+1 == max_images:
            break
            
    print(f"{i+1}/{len(test_images_list)}")
    sc_all = scores_all.get_pr_curves()
    sc_diff = scores_diff.get_pr_curves()
    sc_easy = scores_easy.get_pr_curves()

    write_to_csv_pr_curve(sc_all, iou_threshold, name = "all_pr_curve")
    write_to_csv_pr_curve(sc_diff, iou_threshold, name = "diff_pr_curve")
    write_to_csv_pr_curve(sc_easy, iou_threshold, name = "easy_pr_curve")
    
    duration = time.time() - start_time
    print(f"Duration: {round(duration/60)} minutes")
    print("DONE")
    
    return sc_all, sc_diff, sc_easy

def calculate_AP(precision_recall_dict, name = "AP"):
    average_precision_dict = {}
    
    for cls, threshold_dict in precision_recall_dict.items():
        precisions = []
        recalls = []
            
        for threshold, metric_dict in threshold_dict.items():
            precision = metric_dict["precision"]
            recall = metric_dict["recall"]
            precisions.append(precision)
            recalls.append(recall)

        sorted_indices = np.argsort(recalls)
        sorted_recalls = np.array(recalls)[sorted_indices]
        sorted_precisions = np.array(precisions)[sorted_indices]
    
        average_precision = np.trapz(sorted_precisions, sorted_recalls)
        average_precision_dict[cls] = average_precision


    save_dir = join(RESULT_DIR, "Average_Precision")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_file = os.path.join(save_dir, name + ".csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Average Precision'])
        for key, value in average_precision_dict.items():
            writer.writerow([key, value])
    
    return average_precision_dict

def calculate_mAP(AP_dict):
    sum_ap = 0
    class_counter = 0
    for cls, ap in AP_dict.items():
        class_counter += 1
        sum_ap += ap
    mAP = sum_ap/class_counter
    return mAP

# SCRIPT starts here

model = YOLO(model_path_arg)

pr_all, pr_easy, pr_hard = calculate_precision_recall(
    model, 
    conf_threshold = conf_arg, 
    iou_threshold = iou_arg, 
    save_interval = save_interval_arg,
    max_images = max_img_arg)

sc_all, sc_diff, sc_easy = calculate_precision_recall_curves(
    model, 
    test_images_list, 
    test_targets_list,
    iou_threshold = iou_arg,
    save_interval = save_interval_arg,
    sanity_check = [False, False, False], #all hard easy
    classes_sanity_check = ["large-vehicle"],
    max_images = max_img_arg)

AP_all = calculate_AP(sc_all, name = "AP_all")
AP_diff = calculate_AP(sc_diff, name = "AP_diff")
AP_easy= calculate_AP(sc_easy, name = "AP_easy")

mAP_all = calculate_mAP(AP_all)
print(f"mAP for all instances: {round(100*mAP_all,2)}%")
mAP_diff = calculate_mAP(AP_diff)
print(f"mAP for hard to detect instances: {round(100*mAP_diff,2)}%")
mAP_easy = calculate_mAP(AP_easy)
print(f"mAP for easy instances: {round(100*mAP_easy,2)}%")





![Python](https://img.shields.io/badge/python-3.12.5-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)
# Code

Here you can find the files necessary to replicate our findings. For some files we created Jupyter notebooks for better unterstanding of the code. Before you execute the code, make sure you have the correct Python and PyTorch version and recreate the virtual environment:

```bash
pip install -r req.txt
```
**Overview**
- Use `preprocessing.py` to prepare DOTA for training, notebook: `dataloader.ipynb`
- Use `prepare_yolo.py` to adjust the preprocessed dataset to YOLO standard, notebook: `Yolo_v5.ipynb`
- Use `evaluation.py` to evaluate a model, notebook: `yolo_evaluation.ipynb`
- Use `oversample.py` to oversample a dataset with images containing hard to detect instances, notebook: `experiments.ipynb`
- Use `copy_htd_instances.py` to copy and paste all hard to detect instances in oversampled images, notebook: `experiments.ipynb`
- For a thorough EDA for hard to detect instances in DOTA, refer to the notebook `EDA_hard.ipynb`

## General Scripts

### preprocessing.py

This script preprocesses the DOTA dataset for training by splitting images and their corresponding bounding boxes into quadratic patches of size 1024 with an overlap of 256. Smaller images are zero padded. The processed patches are saved to a `patches_<tile_size>` directory within the specified data directory. For the DOTA dataset there are no public test targets available, which is why a split of the validation set will be used as test data. In the notebook `dataloader.ipynb` the dataset can be easily visualized.

#### Dataset Directory Structure

To use the script, ensure that your data directory is structured as follows:

```bash
├── Data directory
        ├── images_train
        ├── images_val
        ├── labels_train
        ├── labels_val
```

#### Setup

1. **Download the Dataset**: Place the dataset files in the appropriate subdirectories within the data directory.

#### Usage
To run the preprocessing script, use the following command:

  ```bash
  python preprocessing.py --data_dir "path/to/data/directory" --test_size <test_size> --tile_size <tile_size> --overlap <overlap> 
```

| Parameter    | Default Value | Function                                                                 |
|--------------|:---------------:|--------------------------------------------------------------------------|
| data_dir    | "../04-Data"  | Path to the data directory containing the images and annotations. Replace "path/to/data/directory" with the path to your data directory.|
| test_size   | 0.6  | Specifies the ratio of val images and targets which will be allocated to the test set. Note: The ratio refers to the raw DOTA images, i.e. after preprocessing the raw images, the ratio will be slightly off due to different image sizes.|
| tile_size   | 1024  | Size of the quadtratic patches extracted from the original DOTA images and targets.|
| overlap   | 256  | Overlap between the extracted patches. This only applies to the train and validation set, the test set does not have overlap between patches in order to not inflate testing metrics by counting objects twice.|

### prepare_yolo.py

The `prepare_yolo.py` script is designed to prepare the DOTA dataset for YOLO models. This preparation follows the preprocessing of the DOTA dataset, which is handled by `preprocessing.py`. The script converts bounding box annotations into the YOLO format and sets up the necessary directory structure within a `dota_yolo` folder inside the specified data directory. The notebook in `Yolo_v5.ipyn` shows the preparation process in more detail.

#### Setup

1. **Preprocess the DOTA dataset**: Ensure that the DOTA dataset is preprocessed by running `preprocessing.py` first.

#### Usage

To execute the `prepare_yolo.py` script, use the following command:

```bash
python prepare_yolo.py --patch_dir "path/to/patch/directory" --name_dota_dir "name_of_dota_directory"
```
| Parameter    | Default Value | Function                                                                 |
|--------------|:---------------:|--------------------------------------------------------------------------|
| patch_dir    | "../04-Data/patches_1024"  | Replace "path/to/patch/directory" with the path to the patch directory created by `preprocessing.py`, `oversample.py` or `copy_htd_instances.py`.|
| name_dota_dir   | "dota_yolo"  | Specifies the directory where the dota dataset and yaml file is saved to. Replace "name_of_dota_directory" with the name of the dota directory which will be created and later used for training.|

### evaluation.py

The `evaluation.py` script evaluates a given model on various metrics related to object detection. The notebook `yolo_evaluation.ipynb` shows the evaluation process in great detail.

#### Metrics

The script computes the following metrics for all classes and difficulty levels:
- **Precision**: Calculated for a specified IoU threshold and confidence threshold.
- **Recall**: Calculated for a specified IoU threshold and confidence threshold.
- **F1 Score**: Calculated for a specified IoU threshold and confidence threshold.
- **Precision-Recall Curve**: Generated for a specified IoU threshold.
- **Average Precision (AP)**: Computed for a specified IoU threshold.
- **Mean Average Precision (mAP)**: Computed for a specified IoU threshold.

#### Setup

1. **Train** an object detection model
2. **Prediction output format** must be as torch tensor of shape `torch.Size([number_of_predictions, 6])`. The tensor should follow this structure:

   | x_min | y_min | x_max | y_max | confidence | class |
   |-------|-------|-------|-------|------------|-------|
   | float | float | float | float | float      | int   |

   - **x_min**: The minimum x-coordinate of the bounding box (left).
   - **y_min**: The minimum y-coordinate of the bounding box (top).
   - **x_max**: The maximum x-coordinate of the bounding box (right).
   - **y_max**: The maximum y-coordinate of the bounding box (bottom).
   - **confidence**: A float representing the confidence score of the prediction, ranging from 0 to 1.
   - **class**: An integer representing the predicted class label.

   **Example:**

   ```python
   tensor([
       [x_min1, y_min1, x_max1, y_max1, confidence1, class1],
       [x_min2, y_min2, x_max2, y_max2, confidence2, class2],
       ...
   ])
        ```

#### Usage

To execute the `evaluation.py` script, use the following command:

```bash
evaluation.py --patch_dir "path/to/patch/directory" --model_path "path/to/model" --iou <iou_threshold> --conf <confidence_threshold> --max_img <maximum number of images to process> --save_interval <interval for saving data> --result_dir "path/to/results/directory"
```

Explanation of Arguments:

| Parameter    | Default Value | Function                                                                 |
|--------------|:---------------:|--------------------------------------------------------------------------|
| patch_dir     | "../04-Data/patches_1024"  | Path to the patch directory created by `preprocessing.py`. Thereby you can evaluate yolo models on datasets with different image sizes. Replace "path/to/patch/directory" with the path to the patch directory created by `preprocessing.py`.|
| model_path   | "../best.pt"  | Path to the saved model. Replace "path/to/model" with the path to your model.   |
| iou          | 0.5           | IoU threshold to determine valid predictions for precision-recall calculations. |
| conf         | 0.5           | Confidence threshold to determine valid predictions for precision-recall calculations.|
| max_img      | -1            | Maximum number of images to process.                                     |
| save_interval| 100           | Interval for saving evaluation data.                                     |
| result_dir| "Results"           | Directory to save evaluation results in. Replace "path/to/results/directory" with the path to the desired results directory.  |

## Experiment 1

The first experiment in order to increase the performance on hard to detect images is to oversample images with hard to detect images. This approach was taken from the paper [Augmentation for small object detection](https://arxiv.org/pdf/1902.07296). Therefore the script `oversample.py` was created, which prepares experiment one. The notebook in `experiments.ipynb` shows the experimental process in more detail.

### oversample.py
This script copies images with hard to detect objects a specified amount of times, thereby increasing the ratio of images with hard to detect objects in the augmented dataset.

#### Setup

1. **Preprocess the DOTA dataset**: Ensure that the DOTA dataset is preprocessed by running `preprocessing.py` first.

#### Usage

The script takes a patch directory created by `preprocessing.py`, augments it and saves the new dataset to the same parent directory. Only the training set is oversampled, the validation and test set stay the same. The resulting augmented patch directory has the same structure as the non-augmented on and can be prepared for training by `prepare_yolo.py` the same way. To execute the `oversample.py` script, use the following command:

```bash
python oversample.py --patch_dir "path/to/patch/directory" --osf <oversampling factor>"
```
| Parameter    | Default Value | Function                                                                 |
|--------------|:---------------:|--------------------------------------------------------------------------|
| patch_dir    | "../04-Data/patches_1024"  | Directory containing the patches to be oversampled. Replace "path/to/patch/directory" with the path to the patch directory created by `preprocessing.py`.|
| osf   | 0  | The oversampling factor is an integer that determines how many times an image containing hard to detect instances is copied.|

## Experiment 2

The second experiment in order to increase the performance on hard to detect images is to copy hard to detect instances in the copies of the images. The script is able to process patch directories created by `oversample.py`, which oversamples the training set with images containing hard to detect objects. The copies created in this process are augmented by `copy_htd_instances.py` by copying every hard to detect instance once and paste it to a random spot (non-overlapping to other bounding boxes). Thereby the amount of hard to detect instances in the images increases what can increase detection performance of these instances. This approach was taken from the paper [Augmentation for small object detection](https://arxiv.org/pdf/1902.07296). Therefore the script `copy_htd_instances.py` was created, which prepares experiment two. The notebook in `experiments.ipynb` shows the experimental process in more detail.

### copy_htd_instances.py
The script copies hard to detect instances in copies of the original images and pastes them with no overlap to other bounding boxes at a random place in the copied image.

#### Setup

1. **Preprocess the DOTA dataset**: Ensure that the DOTA dataset is preprocessed by running `preprocessing.py` first.

2. **Create an oversampled patch directory**: Run `oversample.py` on a preprocessed DOTA dataset in order to get an oversampled patch directory. This will be used for augmentation

#### Usage

The script takes a patch directory created by `oversample.py`, augments it and saves the new dataset to the same parent directory. Only the hard to detect instances in copied/oversampled images in the training set are copied, the validation and test set stay the same. The resulting augmented patch directory has the same structure as the non-augmented on and can be prepared for training by `prepare_yolo.py` the same way. To execute the `copy_htd_instances.py` script, use the following command:

```bash
python copy_htd_instances.py --patch_dir "path/to/patch/directory"
```
| Parameter    | Default Value | Function                                                                 |
|--------------|:---------------:|--------------------------------------------------------------------------|
| patch_dir    | "../04-Data/patches_1024_os_1"  | Directory containing the patches whose hard to detect instances are to be copied. Replace "path/to/patch/directory" with the path to the patch directory created by `oversample.py`.|












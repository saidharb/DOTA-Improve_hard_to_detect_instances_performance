# Final Project:  Improving object detection performance on hard to detect instances in DOTA

## Content
The goal of the final project is to improve object detection performance on hard to detect instances in the DOTA dataset. For that we will take the approaches of the paper [Augmentation for small object detection](https://arxiv.org/abs/1902.07296) and apply them. Therefore we will train three YOLOv5n models: One baseline model and two experiments. In the first experiment images containing hard to detect instances are oversampled in the dataset. In the second experiment hard to detect instances are copied and pasted within the oversampled images. Finally, both approaches will be compared to the baseline model which was trained on the original dataset and a conlusion is drawn.

## Abstract

Research on object detection continues to evolve, revealing various challenges that persist within the field. One significant challenge is the detection of small instances. In this study, we employ strategies to augment the DOTA dataset to improve object detection performance on hard to detect labeled instances. In particular, we oversample the dataset with images containing hard to detect instances and we copy
and paste hard to detect instances within the oversampled images. Our findings demonstrate that both methods increase the mean average precision on hard to detect instances by 42.2% and 21.4% respectively. In addition the performance on easy to detect instances rises simultaneously.

<details>
  <summary style="font-size:140px">Guidelines</summary>

The detailed explanation on how to prepare the dataset and run the code to recreate our results can be found in the readme file in the [Code](KORREKTUR) directory.

</details>

<details>
  <summary style="font-size:140px">Dataset</summary>
  
The dataset used in our experiments is the [Dota Dataset](https://captain-whu.github.io/DOTA/). In particular we use the version 1.5 of this dataset. It consists of 2.806 Aerial images with different sizes. The image sizes vary from 800x800 to 20.000 x 20.000 pixels. On these images there are 16 classes labeled which are:

- large vehicle
- small vehicle
- helicopter
- plane
- ship
- swimmingpool
- container crane
- storage tank
- bridge
- harbor
- roundabout
- baseball-diamond
- basketball court
- ground track field
- tennis court
- soccerball field

</details>

<details>
  <summary style="font-size:140px">Model</summary>
The model employed is the YOLOv5n. The newest version and information about the model can be found here: 

[Link](https://github.com/ultralytics/yolov5)
  
</details>

<details>
  <summary style="font-size:140px">Results</summary>

Both employed approaches improved the detection performance on hard to detect instances considerably. However the oversampling apporach prooved to be slightly better, as the metrics are the best for this experiment. Refer to the Results section in our report for a detailed analysis.
  
</details>

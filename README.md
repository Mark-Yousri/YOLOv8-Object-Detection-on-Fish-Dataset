# YOLOv8-Object-Detection-on-Fish-Dataset
his notebook demonstrates how to use YOLOv8, a state-of-the-art object detection model, to detect fish, jellyfish, sharks, and tuna in images. The model is trained on a custom dataset of 696 images, using the Keras CV library.

# Dataset
The dataset is taken from the Fall 2023 Intro to Vision Dataset Kaggle competition. It contains 696 images of four classes: fish, jellyfish, shark, and tuna. Each image has a corresponding annotation file that contains the bounding box coordinates for each object in the image.

The dataset is split into train and validation sets, with 556 and 140 images respectively. The images are resized and augmented using the Keras CV library, which provides various image processing and augmentation layers.

# Model
The model is based on the YOLOv8 architecture, which is a single-stage object detector that uses a backbone network, a feature pyramid network (FPN), and a detection head. The backbone network is responsible for extracting features from the input image, the FPN is responsible for aggregating features from different scales, and the detection head is responsible for predicting the class and location of each object.

The backbone network used in this notebook is the YOLOv8-L Backbone, which is a large-scale network that consists of 24 convolutional layers and 3 transition layers. The backbone network is initialized with the weights pretrained on the COCO dataset.

The FPN used in this notebook is a simple one-layer FPN that concatenates the features from the last three transition layers of the backbone network. The detection head used in this notebook is a single-layer convolutional layer that predicts the class and location of each object using a 3x3 kernel.

The model is compiled and trained using the Keras CV library, which provides various utilities and losses for object detection. The model is trained for 60 epochs, using the Nadam optimizer, the binary crossentropy loss for classification, and the complete IoU (CIoU) loss for bounding box regression.

# Results
The model achieves a validation loss of 3.7683, a validation box loss of 2.7745, and a validation class loss of 0.9938. The model is able to detect fish, jellyfish, sharks, and tuna in images with reasonable accuracy and precision.

# Dataset link 
www.kaggle.com/datasets/markyousri/marine-life-classification-dataset

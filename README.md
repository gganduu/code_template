# code_template
This project is to create a code template for image classification, object detection and segmentation.
It's based on pytorch API, it supports to change the pretrained backbone on given config yaml file.

# contents:
    README ...................  This file
    config ...................  directory contains config yaml files
    core   ...................  directory contains classification/detection/segmentation part code and dataset.py
    utils  ...................  directory contains some common functions and detection specific functions
    data   ...................  directory contains your customerized data, it should be organized by specific format
    logs   ...................  directory contains log file
    models ...................  directory contains best models
    
# environment
* python 3.x
* intel-ipex 2.0
* pytorch 1.7

#training dataset
This code is verified by hymenoptera_data and pascal_voc2007

# example
### training
python -c config/resnet50.yml
python -c config/resnet50.yml --ipex=True
### inference
python -c config/resnet50.yml -d path of your data
python -c config/resnet50.yml -d path of your data --ipex=True

# tf-trt-ssd
Very simple TF-TRT application to detect objects from camera captured image

## Prerequisites
1. Install [NVIDIA-AI-IOT/tf_trt_models](https://github.com/NVIDIA-AI-IOT/tf_trt_models) and run [tf_trt_models/examples/detection/detection.ipynb](https://github.com/NVIDIA-AI-IOT/tf_trt_models/blob/master/examples/detection/detection.ipynb) to have ssd_inception_v2_coco_trt.pb which is the deep neural network model for this application.
1. Download [coco-labels-paper.txt](https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt) from [amikelive/coco-labels](https://github.com/amikelive/coco-labels).

## Usage
Locate tf-trt-ssd.py, ssd_inception_v2_coco_trt.pb and coco-labels-paper.txt to the same directory.
'''
$ python3 tf-trt-ssd.py
'''

# MIT License
#
# Copyright (c) 2019 tsutof
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 

from __future__ import print_function

import cv2
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import ast

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960
GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx \
    ! videoconvert \
    ! appsink' % (FRAME_WIDTH, FRAME_HEIGHT)
WINDOW_NAME = 'TF-TRT Object Detection'
MODEL_FILE = './ssd_inception_v2_coco_trt.pb'
LABEL_FILE = './coco-labels-paper.txt'

def load_graph_def(file):
    with tf.gfile.GFile(file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def load_labels(file):
    labels = ['unlabeled']
    with open(file, 'r') as f:
        for line in f.read().splitlines():
            labels.append(line)
    return labels

def main():
    '''
    A simple python application to detect objects from camera captured image
    using TF-TRT for NVIDIA Jetson Nano Developer Kit.
    This application assumes the TensorRT optimized ssd_mobilenet_v1_coco model.
    Refer to the NVIDIA-AI-IOT/tf_trt_models GitHub ripository for details on 
    the model.
    '''

    labels = load_labels(LABEL_FILE)
    num_labels = len(labels)

    print('Loading graph definition...', end = '', flush = True)
    trt_graph_def = load_graph_def(MODEL_FILE)
    print('Done.')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config = tf_config)
    print('Importing graph definition to TensorFlow...', \
        end = '', flush = True)
    tf.import_graph_def(trt_graph_def, name = '')
    print('Done.')

    input_names = ['image_tensor']
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    print('Configuring camera...', end = '', flush = True)
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    print('Done.')

    while True:
        ret, img = cap.read()
        if ret != True:
            break

        # Caputure frame
        imgConv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        imgRs = cv2.resize(imgConv, (300, 300))

        # Do inference
        scores, boxes, classes, num_detections \
        = tf_sess.run( \
            [tf_scores, tf_boxes, tf_classes, tf_num_detections], \
            feed_dict={tf_input: imgRs[None, ...]})

        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]

        for i in range(int(num_detections)):
            # Look up label string
            class_id = int(classes[i])
            label = labels[class_id] if class_id < num_labels else 'unlabeled'

            # Get score
            score = scores[i]

            # Draw bounding box
            box = boxes[i] * np.array( \
                [FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH])
            box = box.astype(np.int)
            cv2.rectangle(img, \
                (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)
            
            # Put label near bounding box
            inf = '%s: %f' % (label, score)
            print(inf)
            cv2.putText(img, inf, (box[1], box[2]), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # Show image
        cv2.imshow(WINDOW_NAME, img)

        # Check if user hits ESC key to exit
        key = cv2.waitKey(1)
        if key == 27: # ESC 
            break

if __name__ == "__main__":
    main()

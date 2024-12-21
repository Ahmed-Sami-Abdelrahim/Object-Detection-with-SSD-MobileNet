Object Detection with SSD MobileNet
This project implements object detection using OpenCV's DNN module with a pre-trained SSD MobileNet V3 model. The model is capable of detecting multiple object classes (such as people, cars, and animals) in real-time from either images or video streams.

Features
Real-time object detection using a webcam or video input.
Bounding boxes and labels are drawn around detected objects.
Pre-trained SSD MobileNet V3 model with COCO dataset class labels.
OpenCV's DNN module for efficient inference.
Requirements
Python 3.x
OpenCV 4.x or higher
Install Dependencies
You can install the required dependencies using pip:

bash
نسخ الكود
pip install opencv-python opencv-python-headless
نسخ الكود
Model Details
This project uses the following files:

coco.names: A file containing the list of class labels (e.g., 'person', 'car', 'dog') that the model can detect.
frozen_inference_graph.pb: The pre-trained weights file for the SSD MobileNet V3 model.
ssd_mobilenet_v3_large_coco.pbtxt: The configuration file describing the structure of the SSD MobileNet V3 model.
Usage
Running Object Detection from Video
To run the object detection script using your webcam or any video source, execute the following command:

bash
نسخ الكود
python main.py
This will start the video capture from your webcam and perform object detection on each frame.

Running Object Detection on an Image
To use an image file, modify the script to read an image instead of video input, and use OpenCV functions to display or save the results.

Code Explanation
Initialization
The model is initialized with the pre-trained weights (frozen_inference_graph.pb) and the configuration (ssd_mobilenet_v3_large_coco.pbtxt).
Class names are loaded from the coco.names file to identify the detected objects.
Processing Video
The video capture (cv2.VideoCapture(0)) starts from the webcam.
For each frame, the dnn.detect() method is used to detect objects in the frame.
Bounding boxes and class labels are drawn around the detected objects.
Termination
Press q to stop the video capture and exit the program.

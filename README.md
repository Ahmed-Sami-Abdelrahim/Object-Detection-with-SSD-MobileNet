# Object Detection with SSD MobileNet

This project implements real-time object detection using OpenCV's DNN module with a pre-trained SSD MobileNet V3 model. The model can detect multiple object classes (e.g., people, cars, animals) from either images or video streams.

## Features

- **Real-time Object Detection**: Detects objects in video frames captured from a webcam or video file.
- **Bounding Boxes**: Draws bounding boxes around detected objects.
- **Class Labels**: Labels each detected object with its corresponding class name (from COCO dataset).
- **Pre-trained Model**: Uses a pre-trained SSD MobileNet V3 model trained on the COCO dataset.
- **OpenCV DNN Module**: Efficient and optimized inference with OpenCV's deep learning (DNN) module.
- **Video and Image Support**: Detect objects from live video streams or static images.

## Requirements

This project requires Python 3.x and the following libraries:

- OpenCV 4.x or higher


## Model Details

This project uses the following files:

- **`coco.names`**: A file containing the list of class labels that the SSD MobileNet V3 model can detect (e.g., 'person', 'car', 'dog').
- **`frozen_inference_graph.pb`**: The pre-trained model weights used for the object detection task.
- **`ssd_mobilenet_v3_large_coco.pbtxt`**: The configuration file describing the structure of the SSD MobileNet V3 model.

## Code Explanation

### <span style="font-weight: bold; color: red;">Model Initialization</span>

- **Pre-trained Weights and Config**: The pre-trained SSD MobileNet V3 model is loaded using `cv2.dnn_DetectionModel` with the weight and configuration files (**`frozen_inference_graph.pb`** and **`ssd_mobilenet_v3_large_coco.pbtxt`**).
- **Class Names**: The class names (from COCO dataset) are loaded from the **`coco.names`** file.

### <span style="font-weight: bold; color: red;">Video Processing</span>

- **Capture Video**: The webcam feed is captured using OpenCV’s `cv2.VideoCapture(0)`.
- **Object Detection**: For each frame, the `dnn.detect()` method is used to detect objects and get their bounding boxes.
- **Draw Bounding Boxes**: The detected bounding boxes and class labels are drawn on the frame.

### <span style="font-weight: bold; color: red;">Termination</span>

- Press **`q`** to stop the video capture and exit the program.


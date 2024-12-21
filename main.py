import cv2

# Paths to class labels, model config, and pre-trained weights for object detection.
class_names = []
class_file = 'coco.names'
config_path = 'ssd_mobilenet_v3_large_coco.pbtxt'
weights_path = 'frozen_inference_graph.pb'

# Load class names
with open(class_file, 'r') as file:
    class_names = file.read().strip().split('\n')

# Initialize the model
dnn_net = cv2.dnn_DetectionModel(weights_path, config_path)
dnn_net.setInputSize(320, 230)  # Set input image size
dnn_net.setInputScale(1.0 / 127.5)  # Normalize image
dnn_net.setInputMean((127.5, 127.5, 127.5))  # Subtract mean
dnn_net.setInputSwapRB(True)  # Convert image from BGR to RGB

def process_video():
    """
    Process live video stream (from webcam) to detect objects.
    """
    cap = cv2.VideoCapture(0)

    # Set video frame resolution
    cap.set(3, 740)  # Width
    cap.set(4, 580)  # Height

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Object detection on the video frame
        class_ids, confidences, bboxes = dnn_net.detect(frame, confThreshold=0.5)
        print(class_ids, bboxes)

        # Draw bounding boxes and labels
        if len(class_ids) > 0:
            for class_id, confidence, bbox in zip(class_ids.flatten(), confidences.flatten(), bboxes):
                color = (0, 255, 0) 
                thickness = 2  
                cv2.rectangle(frame, tuple(bbox), color=color, thickness=thickness)

                # Label and draw the background for the label
                label = class_names[class_id - 1]
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_x = bbox[0]
                label_y = bbox[1] - 10  # Above the bounding box

                # Draw the background rectangle for the label
                cv2.rectangle(frame, (label_x, label_y), (label_x + label_size[0], label_y - label_size[1] - baseline), color=color, thickness=cv2.FILLED)

                # Add the label text on top of the background
                text_color = (255, 255, 255)  # White color for the text
                cv2.putText(frame, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Show the frame with improved bounding boxes and labels
        cv2.imshow('Video Output', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call video processing function
process_video()

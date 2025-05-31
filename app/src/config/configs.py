# Importing necessary libraries
import cv2
import os

# //=======================================//
# Specify the image file to be used.
image = "pic1.jpg"
# image = "pic2.jpg"
# image = "pic3.jpg"

# Specify the video file to be used.
video = "video1.mp4"        # 3840x2160    30fps
# video = "video2.mp4"      # 1920x1080    30fps

# Define the base path for resources (images and videos).
# path = 'D:/dataset/'
# path = './resources/'
path = os.getcwd()


# Construct the full paths for the image and video files
imagepath = os.path.join(path, "app", "resources", "image", image)
imagepath = "-"     # Uncomment this line to disable image input
videopath = os.path.join(path, "app", "resources", "video", video)

# Define the video source
# s = "http://192.168.30.139:4747/video"
# s = "http://192.168.30.139:8080/video"
# s = 0             # webcam
# s = 1             # external webcam
s = videopath       # video file

# //=======================================//
# Paths to the model configuration, weights, and class labels
# modelpath should point to a pre-trained model file (e.g., .weights, .onnx, or .caffemodel).
# configpath should point to the corresponding configuration file (e.g., .cfg or .prototxt).

configpath = os.path.join(os.getcwd(), "app", "models", "saved_model", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelpath = os.path.join(os.getcwd(), "app", "models", "saved_model", "frozen_inference_graph.pb")
classespath = os.path.join(os.getcwd(), "app", "models", "saved_model", "coco.names")

# Uncomment the following lines to switch to a different model or class labels
# configpath = os.path.join(os.getcwd(), "app", "models", "saved_model0", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
# modelpath = os.path.join(os.getcwd(), "app", "models", "saved_model0", "frozen_inference_graph.pb")
# classespath = os.path.join(os.getcwd(), "app", "models", "saved_model0", "coco.txt")


# //=======================================//
# Font settings for displaying text on the video frames
FONT_SIZE_FPS = 2             # Font size for FPS display
FONT_SIZE_TXT = 1             # Font size for general text
FONT_SIZE_OBJ = 1             # Font size for object labels

FONT_COLOR_FPS = (0, 255, 0)    # Green color for FPS text
FONT_COLOR_TXT = (0, 255, 0)    # Green color for general text
FONT_COLOR_OBJ = (255, 0, 255)  # Magenta color for object labels

FONT_THICKNESS_FPS = 3          # Thickness of FPS text
FONT_THICKNESS_TXT = 2          # Thickness of general text
FONT_THICKNESS_OBJ = 2          # Thickness of object labels

FONT_FPS = cv2.FONT_HERSHEY_SIMPLEX   # Font style for FPS text
FONT_TXT = cv2.FONT_HERSHEY_SIMPLEX   # Font style for general text
FONT_OBJ = cv2.FONT_HERSHEY_COMPLEX   # Font style for object labels

FONT_POS_INT = 100           # Position for displaying text on the video frame
FONT_SPACE_TXT = 50          # Space between lines of text for general text

# cv2.FONT_HERSHEY_SIMPLEX
# cv2.FONT_HERSHEY_COMPLEX
# cv2.FONT_HERSHEY_PLAIN
# cv2.FONT_HERSHEY_DUPLEX

# //=======================================//
# Configuration settings for the object detection model
# model_settings:
#   model_path: "models/pretrained_model/saved_model" # Path to the saved model
#   confidence_threshold: 0.5                         # Confidence threshold for detections
#   nms_threshold: 0.4                                # Non-Maximum Suppression (NMS) threshold
#   # input_size: [320, 320]                            # Input size for the model (width, height)
#   input_size: [640, 480]                            # Input size for the model (width, height)
#   labels: "models/pretrained_model/labels.txt"      # Path to the labels file (if applicable)

# video_stream:
#   source: 0                                         # Video source (0 for webcam, or path to video file)
#   width: 640                                        # Width of the video stream
#   height: 480                                       # Height of the video stream
#   fps: 30                                           # Frames per second

# voice_command:
#   language: "en-US"                                 # Language for voice recognition
#   sensitivity: 0.8                                  # Sensitivity for voice commands


# //=======================================//
# save_model.py:
#   --weights: path to weights file
#     (default: './data/yolov4.weights')
#   --output: path to output
#     (default: './checkpoints/yolov4-416')
#   --[no]tiny: yolov4 or yolov4-tiny
#     (default: 'False')
#   --input_size: define input size of export model
#     (default: 416)
#   --framework: what framework to use (tf, trt, tflite)
#     (default: tf)
#   --model: yolov3 or yolov4
#     (default: yolov4)

#  object_tracker.py:
#   --video: path to input video (use 0 for webcam)
#     (default: './data/video/test.mp4')
#   --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
#     (default: None)
#   --output_format: codec used in VideoWriter when saving video to file
#     (default: 'XVID)
#   --[no]tiny: yolov4 or yolov4-tiny
#     (default: 'false')
#   --weights: path to weights file
#     (default: './checkpoints/yolov4-416')
#   --framework: what framework to use (tf, trt, tflite)
#     (default: tf)
#   --model: yolov3 or yolov4
#     (default: yolov4)
#   --size: resize images to
#     (default: 416)
#   --iou: iou threshold
#     (default: 0.45)
#   --score: confidence threshold
#     (default: 0.50)
#   --dont_show: dont show video output
#     (default: False)
#   --info: print detailed info about tracked objects
#     (default: False)

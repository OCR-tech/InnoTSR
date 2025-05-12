# Importing necessary libraries
import cv2
import os

# //=======================================//
# Specify the image file to be used.
image = "pic1a.jpg"
# image = "pic2a.jpg"
# image = "pic3a.jpg"

# Specify the video file to be used.
video = "street1.mp4"      # street        # 3840x2160    24fps
# video = "street2.mp4"     # bike          # 1920x1080    30fps
# video = "street3.mp4"     # bike/night    # 1920x1080    30fps

# Define the base path for resources (images and videos).
# path = 'D:/dataset/'
# path = './resources/'
path = os.getcwd()

# Construct the full paths for the image and video files
imagepath = os.path.join(path, "app", "resources", "video", image)
imagepath = "-"
videopath = os.path.join(path, "app", "resources", "video", video)

# Uncomment the following lines to debug and print the paths
# print("imagepath := ", imagepath)
# print("videopath := ", videopath)

# Define the video source
# The source can be:
# - Integer for a webcam (e.g., 0 for the default webcam, 1 for an external webcam)
# - Path to a video file
# - URL for an IP camera stream
# s = "http://192.168.30.139:4747/video"
# s = "http://192.168.30.139:8080/video"
# s = 0
# s = 1
s = videopath

# //=======================================//
# Paths to the model configuration, weights, and class labels
# modelpath should point to a pre-trained model file (e.g., .weights, .onnx, or .caffemodel).
# configpath should point to the corresponding configuration file (e.g., .cfg or .prototxt).

configpath = os.path.join(os.getcwd(), "app", "models", "saved_model", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelpath = os.path.join(os.getcwd(), "app", "models", "saved_model", "frozen_inference_graph.pb")
classespath = os.path.join(os.getcwd(), "app", "models", "saved_model", "coco.names")
# print("configpath := ", configpath)
# print("modelpath := ", modelpath)
# print("classespath := ", classespath)

# Uncomment the following lines to switch to a different model or class labels
# configpath = os.path.join("models/saved_model", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
# modelpath = os.path.join("models/saved_model", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
# classespath = os.path.join("models/saved_model", "coco_class_labels.txt")

# //=======================================//
# Font settings for displaying text on the video frames
FONT_SIZE_FPS = 0.8             # Font size for FPS display
FONT_SIZE_TXT = 0.43            # Font size for general text
FONT_SIZE_OBJ = 0.5             # Font size for object labels

FONT_COLOR_FPS = (0, 255, 0)    # Green color for FPS text
FONT_COLOR_TXT = (0, 255, 0)    # Green color for general text
FONT_COLOR_OBJ = (255, 0, 255)  # Magenta color for object labels

FONT_THICKNESS_FPS = 2          # Thickness of FPS text
FONT_THICKNESS_TXT = 1          # Thickness of general text
FONT_THICKNESS_OBJ = 1          # Thickness of object labels

FONT_FPS = cv2.FONT_HERSHEY_SIMPLEX   # Font style for FPS text
FONT_TXT = cv2.FONT_HERSHEY_SIMPLEX   # Font style for general text
FONT_OBJ = cv2.FONT_HERSHEY_COMPLEX   # Font style for object labels

# FONT_TXT = cv2.FONT_HERSHEY_SIMPLEX
# FONT_TXT = cv2.FONT_HERSHEY_COMPLEX
# FONT_TXT = cv2.FONT_HERSHEY_PLAIN
# FONT_TXT = cv2.FONT_HERSHEY_DUPLEX

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

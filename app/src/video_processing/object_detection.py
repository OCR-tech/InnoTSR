import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
# import datetime
# import getpass

from src.voice_processing.voice_output import speak  # Import the speak function for voice output

# Import paths and configurations from other modules
from main import classespath, configpath, modelpath   # Paths for model and class files
from src.config.configs import FONT_FPS, FONT_COLOR_FPS, FONT_SIZE_FPS, FONT_THICKNESS_FPS
from src.config.configs import FONT_FPS, FONT_COLOR_FPS, FONT_SIZE_FPS, FONT_THICKNESS_FPS
from src.config.configs import FONT_TXT, FONT_COLOR_TXT, FONT_SIZE_TXT, FONT_THICKNESS_TXT
from src.config.configs import FONT_OBJ, FONT_COLOR_OBJ, FONT_SIZE_OBJ, FONT_THICKNESS_OBJ
from src.config.configs import FONT_POS_INT, FONT_SPACE_TXT

from src.video_processing.sort import Sort

# Set a random seed for reproducibility
np.random.seed(5)

# Set the time interval for the timer
sec = 1
# sec = 0.5
# sec = 0.1




# #//==================================//
# class TrafficSignDetector1:
#     def __init__(self, model_path):
#         self.model = tf.keras.models.load_model(model_path)
#         self.class_names = ["Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
#                             "Speed Limit 80", "End of Speed Limit 80", "Speed Limit 100", "Speed Limit 120", "No Passing",
#                             "No Passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection",
#                             "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
#                             "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
#                             "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
#                             "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
#                             "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
#                             "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
#                             "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
#                             "End of no passing by vehicles over 3.5 metric tons"]


#     def detect(self, image):
#         image_resized = cv2.resize(image, (64, 64))
#         image_array = np.expand_dims(image_resized, axis=0)
#         predictions = self.model.predict(image_array)
#         class_index = np.argmax(predictions)
#         confidence = predictions[0][class_index]
#         return self.class_names[class_index], confidence

#     def draw_bounding_box(self, image, box, label, confidence):
#         (x, y, w, h) = box
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         text = f"{label}: {confidence:.2f}"
#         cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)






#//==================================//
def command_assistant(detector_instance):
    """
    Command assistant to process voice commands and execute corresponding actions.
    """
    from src.voice_processing.voice_input import listen_for_commands
    from src.voice_processing.voice_output import speak, set_volumn, on_volumn, off_volumn

    print('=== command_assistant ===')
    # detector_instance.timer_alert.stop()    # Stop the alert timer
    detector_instance.timer_assistant.stop()  # Stop the assistant timer

    # Start listening for voice commands
    while True:
        command = listen_for_commands()  # Listen for a voice command
        # command = "exit"
        # command = "start"
        # command = "help"
        # command = "capture"
        # print(f"command: {command}")
        # speak(command)

        if command:
            if command == "exit":
                print("Exiting the system.")
                speak("Exiting the system.")
                on_volumn()
                os._exit(1)
            elif command == "stop":
                print("Stop the system.")
                speak("Stop the system.")
                return
            elif command == "help":
                print("Available commands: capture, pause, resume, stop, record, play, save, alert, exit")
                speak("Available commands: capture, pause, resume, stop, record, play, save, alert, exit")
                return
            elif command == "capture":
                print("Capturing frame...")
                speak("Capturing frame...")
                # capture_frame(detector_instance)
                return
            elif command == "pause":
                print("Pausing frame...")
                speak("Pausing frame...")
                # pause_frame(detector_instance)
                return
            elif command == "resume":
                print("Resuming the system...")
                speak("Resuming the system...")
                # resume_frame(detector_instance)
                return
            elif command == "record":
                print("Recording frame...")
                speak("Recording frame...")
                # record_frame(detector_instance)
                return
            elif command == "play":
                print("Playing frame...")
                speak("Playing frame...")
                return
            elif command == "save":
                print("Save frame...")
                speak("Save frame...")
                cv2.imwrite("snapshot.jpg", detector_instance.image_out)
                return
            elif command == "menu":
                print("Displaying menu")
                speak("Displaying menu")
                return
            elif command == "on":
                print("Alert On!")
                speak("Alert On!")
                on_volumn()
                return
            elif command == "off":
                print("Alert Off!")
                speak("Alert Off!")
                off_volumn()
                return
            else:
                speak("Unknown command")
                print('Unknown command')
                return
        else:
            print('No command detected.')

#//==================================//
def command_alert(detector_instance):
    """
    Command alert to process detected objects and generate voice alerts.
    """
    from src.voice_processing.voice_process import voice_processing

    print('=== command_alert ===')
    detector_instance.timer_assistant.stop()  # Stop the assistant timer
    detector_instance.timer_alert.stop()  # Stop the alert timer

    # Process detected objects and generate voice alerts
    voice_processor = voice_processing(detector_instance.obj)
    detector_instance.voice = voice_processor.voice_alert(detector_instance)

    # Restart the timers
    detector_instance.timer_assistant.start()
    detector_instance.timer_alert.start()

# //===========================================//
def exit_program(detector_instance):
    """
    Exit the program and release resources.
    """
    print("=== exit_program ===")
    detector_instance.timer_assistant.stop()  # Stop the assistant timer
    detector_instance.timer_alert.stop()  # Stop the alert timer
    detector_instance.source.release()  # Release the video source
    cv2.destroyWindow(detector_instance.window)  # Close the OpenCV window
    cv2.destroyAllWindows()  # Close all OpenCV windows
    os._exit(1)  # Exit the program

# //=======================================//
def capture_image_objbox(detector_instance):
    """
    Capture the current frame with object bounding boxes and save it as an image.
    """
    print('=== capture_image_objbox ===')
    # Get the absolute path of the current script
    # absolutepath = os.path.abspath(__file__)
    # fileDirectory = os.path.dirname(absolutepath)  # Directory of the current script
    # parentDirectory = os.path.dirname(fileDirectory)  # Parent directory

    currentworkingDirector = os.getcwd()    # Get the current working directory
    image_path = os.path.join(currentworkingDirector, "app", "data", "saved_image")  # Path to the "image" directory
    print('image_path := ', image_path)

    # Construct the file name for the image with bounding boxes
    image_file = image_path + '\\' + f'image_{detector_instance.frame_count:03d}_obj.jpg'

    # Save the current frame with bounding boxes as an image
    cv2.imwrite(image_file, detector_instance.image_out)
    print('image_save_objbox := ', image_file)

# # //=======================================//
# def detect_objects(net, img):
#     """
#     Detect objects in the given image using the specified neural network.
#     """

#     # print('=== detect_objects ===')

#     # Perform object detection using the neural network
#     classlabelids, confidence, bbox = net.detect(img, confThreshold=0.5, nmsThreshold=0.4)

#     # Convert bounding boxes and confidence scores to lists
#     bbox = list(bbox)
#     confidence = list(np.array(confidence).reshape(1, -1)[0])
#     confidence = list(map(float, confidence))
#     # print('classlabelid := ', classlabelids)
#     # print('confidence := ', confidence)
#     # print('bbox := ', bbox)

#     # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
#     bboxidx = cv2.dnn.NMSBoxes(bbox, confidence, score_threshold=0.5, nms_threshold=0.1)
#     # print('len(bbox) := ', len(bboxidx))
#     # print('bboxidx := ', bboxidx)

#     return bbox, bboxidx, confidence, classlabelids

# //=======================================//
def detect_objects(detector_instance, net, img):
    """
    Detect objects in the given image using the specified neural network.
    """

    # print('=== detect_objects ===')

    # Perform object detection using the neural network
    # classlabelids, confidence, bbox = net.detect(img, confThreshold=0.5, nmsThreshold=0.4)
    classlabelids, confidence, bbox = net.detect(img, confThreshold=0.6, nmsThreshold=0.5)
    # print('classlabelids := ', classlabelids)
    # print('confidence := ', confidence)
    # print('bbox := ', bbox)


    # if classlabelids is match to classlabelids_idx, then remove the corresponding bbox and confidence
    bbox = [bbox[i] for i in range(len(bbox)) if classlabelids[i] not in detector_instance.classlabelids_idx]
    confidence = [confidence[i] for i in range(len(confidence)) if classlabelids[i] not in detector_instance.classlabelids_idx]
    classlabelids = [classlabelids[i] for i in range(len(classlabelids)) if classlabelids[i] not in detector_instance.classlabelids_idx]
    # print('classlabelids1 := ', classlabelids)
    # print('confidence1 := ', confidence)
    # print('bbox1 := ', bbox)


    # Sort the detected objected in x-axis and y-axis respectively, and keep tracking with the same numbering for classlabelids
    sorted_indices = sorted(range(len(bbox)), key=lambda i: (bbox[i][1], bbox[i][0]))
    bbox = [bbox[i] for i in sorted_indices]
    confidence = [confidence[i] for i in sorted_indices]
    classlabelids = [classlabelids[i] for i in sorted_indices]

    # Convert bounding boxes and confidence scores to lists
    bbox = list(bbox)
    confidence = list(np.array(confidence).reshape(1, -1)[0])
    confidence = list(map(float, confidence))

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    bboxidx = cv2.dnn.NMSBoxes(bbox, confidence, score_threshold=0.5, nms_threshold=0.1)
    # print('len(bbox) := ', len(bboxidx))

    bbox1 = []
    confidence1 = []
    classlabelids1 = []

    # loop with the index value inside of bboxidx
    for i in range(len(bboxidx)):
        idx = np.squeeze(bboxidx[i])
        bbox1.append(bbox[idx])
        confidence1.append(confidence[idx])
        classlabelids1.append(classlabelids[idx])

    bbox = bbox1
    confidence = confidence1
    classlabelids = classlabelids1

    # print('=== detect_objects ===')
    # print('bboxidx := ', bboxidx)
    # print('classlabelid := ', classlabelids)
    # print('confidence := ', confidence)
    # print('bbox := ', bbox)

    return bbox, bboxidx, confidence, classlabelids

# //=======================================//
def sort_objects(detector_instance, bbox, bboxidx, confidence, classlabelids):

    # Sort the bounding boxes and class labels based on the indices from NMS (confidence)
    dets = []
    for i in range(len(bboxidx)):
        # idx = np.squeeze(bboxidx[i])
        x, y, w, h = bbox[i]
        score = confidence[i]
        dets.append([x, y, x + w, y + h, score])
    dets = np.array(dets)
    # cast to int
    dets = dets.astype(np.int32)
    # print('dets := ', dets)

    # Update tracker
    tracks = detector_instance.tracker.update(dets)
    tracks = tracks.astype(np.int32)
    # print('tracks := ', tracks)


    # Map track_id to classlabelid
    track_id_to_class = {}
    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track.astype(int)
        # Find the detection that matches this track (by IoU or nearest bbox)
        # Here, we use a simple nearest center approach
        track_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        min_dist = float('inf')
        matched_classlabelid = None
        for j in range(len(bbox)):
            bx, by, bw, bh = bbox[j]
            det_center = np.array([bx + bw / 2, by + bh / 2])
            dist = np.linalg.norm(track_center - det_center)
            if dist < min_dist:
                min_dist = dist
                matched_classlabelid = classlabelids[j]
        track_id_to_class[track_id] = matched_classlabelid
        # print('=== sort_objects ===')
        # print('bbox := ', bbox)
        # print('track_id_to_class := ', track_id_to_class)

    return tracks, track_id_to_class

# # //=======================================//
# def display_objects(detector_instance, img, bbox1, bboxidx, confidence, classlabelids):
#     """
#     Display detected objects on the image with bounding boxes, labels, and confidence scores.
#     Args:
#         detector_instance: The instance containing detection-related data and configurations.
#         img (numpy.ndarray): The input image where objects are detected.
#         bbox1 (list): List of bounding boxes for detected objects.
#         bboxidx (list): Indices of bounding boxes after applying Non-Maximum Suppression (NMS).
#         confidence (list): Confidence scores for detected objects.
#         classlabelids (list): Class label IDs for detected objects.
#     Returns:
#         numpy.ndarray: The image with bounding boxes, labels, and confidence scores drawn.
#     """

#     # print('=== display_objects ===')

#     # Initialize lists to store detection results
#     detector_instance.objid = []        # Object IDs
#     detector_instance.obj = []          # Object labels
#     detector_instance.bbox = []         # Bounding boxes
#     detector_instance.confidence = []   # Confidence scores

#     # Check if there are any valid bounding boxes after NMS
#     if len(bboxidx) != 0:
#         for i in range(0, len(bboxidx)):
#             objid = [i + 1]   # Assign a unique ID to each detected object

#             # Uncomment the following lines for debugging
#             # print('//--------------------------//')
#             # print('i := ', i)
#             # print('id := ', id)
#             # print(len(bboxidx))
#             # print('bboxidx[i] := ', bboxidx[i])

#             # Extract the bounding box for the current object
#             bbox = bbox1[np.squeeze(bboxidx[i])]
#             x, y, w, h = bbox  # Coordinates and dimensions of the bounding box

#             # Extract the confidence score and class label ID
#             classconfidence = np.round(confidence[np.squeeze(bboxidx[i])],2)
#             classlabelid = np.squeeze(classlabelids[np.squeeze(bboxidx[i])])
#             # print('classconfidence := ', classconfidence)
#             # print('classlabelid := ', classlabelid)

#             # Get the class label and color for the current object
#             classlabel = [detector_instance.classeslist[classlabelid]]
#             classcolor = [int(c) for c in detector_instance.colorlist[classlabelid]]
#             # print('classeslist := ', detector_instance.classeslist)
#             # print('classlabel := ', classlabel)
#             # print('classcolor := ', classcolor)

#             # detector_instance.obj = detector_instance.obj + ', ' + classlabel
#             # self.obj = ' '.join(self.obj['label'])
#             # print('list := ', detector_instance.obj)

#             # Format the label to display object ID, class label, and confidence score
#             label = "{}: {}: {:.2f}".format(objid, classlabel, classconfidence)
#             # label = "{}: {:.2f}: {}".format(classlabel, classconfidence, i)
#             # print('Text := ', Text)



#             # Update the detector instance with the detection results
#             detector_instance.objid += objid
#             detector_instance.bbox += [[x, y, w, h]]
#             detector_instance.confidence += [classconfidence]
#             detector_instance.obj += classlabel


#             # # bbox = bbox1[np.squeeze(bboxidx[i])].tolist()
#             # bbox = [[x,y,w,h]]
#             # classconfidence = [classconfidence]

#             # detector_instance.objid = detector_instance.objid + objid
#             # detector_instance.bbox = detector_instance.bbox + bbox
#             # detector_instance.confidence = detector_instance.confidence + classconfidence
#             # detector_instance.obj = detector_instance.obj + classlabel




#             # Put the text on the image
#             # cv2.rectangle(detector_instance.image_resize, (x,y), (x+w, y+h), color=classcolor, thickness=THICKNESS1)
#             # cv2.putText(self.image, classlabel, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
#             # cv2.imshow("Result3", self.image)
#             # cv2.putText(self.image, str(round(classconfidence*100, 2)), (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 1)
#             # cv2.imshow("Result4", self.image)

#             # cv2.putText(img, "Menu Options", (15, 50), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#             cv2.putText(img, "'S': Start", (15, 50), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#             cv2.putText(img, "'M': Menu", (15, 70), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#             cv2.putText(img, "'Esc': Exit", (15, 90), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)

#             if detector_instance.FLAG_MENU:
#                 # Display menu options on the image using putText
#                 # cv2.putText(img, "Press 'S': save", (15, 150), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 # cv2.putText(img, "Press 'H': help", (15, 170), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'+' +Sound", (15, 110), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'-' -Sound", (15, 130), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'1': Capture Image", (15, 150), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'2': Capture Image with Boxes", (15, 170), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'3' Pause", (15, 190), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'4' Resume", (15, 210), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'5' Save", (15, 230), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'6' Record", (15, 250), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'7' Play", (15, 270), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'8' Program", (15, 290), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'9' Help", (15, 310), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#                 cv2.putText(img, "'0' Mute", (15, 330), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
#             else:
#                 pass

#             # Draw the bounding box and label on the image
#             cv2.putText(img, label, (x, y - 10), FONT_OBJ, FONT_SIZE_OBJ, classcolor, FONT_THICKNESS_OBJ)

#             # Draw corner lines for the bounding box
#             linewidth = min(int(w * 0.3), int(h * 0.3))
#             cv2.line(img, (x, y), (x + linewidth, y), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x, y), (x, y + linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x + w, y), (x + w - linewidth, y), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x + w, y), (x + w, y + linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x, y + h), (x + linewidth, y + h), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x, y + h), (x, y + h - linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x + w, y + h), (x + w - linewidth, y + h), classcolor, thickness=FONT_THICKNESS_OBJ)
#             cv2.line(img, (x + w, y + h), (x + w, y + h - linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)

#             # //=======================================//
#             # if  detector_instance.imagepath == "-":
#             #     cv2.putText(img, 'FPS: '+str(int(detector_instance.fps)), (20, 40), FONT, FONT_SCALE2, (255,255,255), THICKNESS1)
#             # # cv2.imshow("Result5", self.image)

#             # cv2.putText(img, f"FPS: {int(detector_instance.fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(img, f"FPS: {detector_instance.fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(img, 'FPS: '+str(int(detector_instance.fps)), (20, 40), FONT, FONT_SCALE2, (255,255,255), THICKNESS1)

#             # self.image1 = img.copy()
#             # cv2.waitKey(1)
#             # time.sleep(5)

#     # #//=====================================
#     # print('objid :=', detector_instance.objid)
#     # print('obj :=', detector_instance.obj)
#     # print('bbox :=', detector_instance.bbox)
#     # print('confidence :=', detector_instance.confidence)
#     # print(detector_instance.bbox[0])
#     # print(detector_instance.bbox[1])

#     # #//=====================================
#     # keys = ["objid", "obj", "objbox", "objconf"]
#     # values = [detector_instance.objid, detector_instance.obj, detector_instance.bbox, detector_instance.confidence]
#     # dict_obj = dict(zip(keys, values))
#     # print(dict_obj)

#     return img

# //=======================================//
def display_objects(detector_instance, img, tracks, track_id_to_class):
    """
    Display detected objects on the image with bounding boxes, labels, and confidence scores.
    Args:
        detector_instance: The instance containing detection-related data and configurations.
        img (numpy.ndarray): The input image where objects are detected.
        tracks (list): List of tracked objects.
        track_id_to_class (dict): Mapping of track IDs to class labels.
    Returns:
        numpy.ndarray: The image with bounding boxes, labels, and confidence scores drawn.
    """

    # print('=== display_objects ===')

    # Initialize lists to store detection results
    detector_instance.objid = []        # Object IDs
    detector_instance.obj = []          # Object labels
    detector_instance.bbox = []         # Bounding boxes
    # detector_instance.confidence = []   # Confidence scores

    # Draw bounding boxes and labels on the image
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        classlabelid = track_id_to_class.get(track_id, None)
        # print('classlabelid := ', classlabelid)

        classlabel = [detector_instance.classeslist[classlabelid]]
        classcolor = [int(c) for c in detector_instance.colorlist[classlabelid]]

        if classlabelid is not None:
            classlabel = detector_instance.classeslist[classlabelid]
            label = f"ID {track_id}: {classlabel}"
        else:
            label = f"ID {track_id}: Unknown"

        # Draw corner lines for the bounding box
        # cv2.rectangle(img, (x1, y1), (x2, y2), classcolor, thickness=FONT_THICKNESS_OBJ)
        w = x2 - x1
        h = y2 - y1
        x = x1
        y = y1
        linewidth = min(int(w * 0.3), int(h * 0.3))
        cv2.line(img, (x, y), (x + linewidth, y), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x, y), (x, y + linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x + w, y), (x + w - linewidth, y), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x + w, y), (x + w, y + linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x, y + h), (x + linewidth, y + h), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x, y + h), (x, y + h - linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x + w, y + h), (x + w - linewidth, y + h), classcolor, thickness=FONT_THICKNESS_OBJ)
        cv2.line(img, (x + w, y + h), (x + w, y + h - linewidth), classcolor, thickness=FONT_THICKNESS_OBJ)

        # Put the label on the image
        cv2.putText(img, label, (x1, y1 - 10), FONT_OBJ, FONT_SIZE_OBJ, classcolor, FONT_THICKNESS_OBJ)
        # cv2.putText(img, label, (x, y - 10), FONT_OBJ, FONT_SIZE_OBJ, classcolor, FONT_THICKNESS_OBJ)

        # Update the detector instance with the detection results
        detector_instance.objid += [track_id]
        detector_instance.bbox += [[x, y, w, h]]
        detector_instance.obj += [classlabel]
        # detector_instance.confidence += [classconfidence]


    # //=======================================//
    # Put the text on the image
    # cv2.rectangle(detector_instance.image_resize, (x,y), (x+w, y+h), color=classcolor, thickness=THICKNESS1)
    # cv2.putText(self.image, classlabel, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    # cv2.imshow("Result3", self.image)
    # cv2.putText(self.image, str(round(classconfidence*100, 2)), (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 1)
    # cv2.imshow("Result4", self.image)


    # cv2.putText(img, "Menu Options", (15, 50), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
    cv2.putText(img, "'S': Start", (15, FONT_POS_INT + 1 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
    cv2.putText(img, "'M': Menu", (15, FONT_POS_INT + 2 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
    cv2.putText(img, "'Esc': Exit", (15, FONT_POS_INT + 3 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)

    if detector_instance.FLAG_MENU:
        # Display menu options on the image using putText
        # cv2.putText(img, "Press 'S': save", (15, 150), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        # cv2.putText(img, "Press 'H': help", (15, 170), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'+' +Sound", (15, FONT_POS_INT + 4 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'-' -Sound", (15, FONT_POS_INT + 5 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'1': Capture Image", (15, FONT_POS_INT + 6 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'2': Capture Image with Boxes", (15, FONT_POS_INT + 7 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'3' Pause", (15, FONT_POS_INT + 8 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'4' Resume", (15, FONT_POS_INT + 9 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'5' Save", (15, FONT_POS_INT + 10 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'6' Record", (15, FONT_POS_INT + 11 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'7' Play", (15, FONT_POS_INT + 12 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'8' Program", (15, FONT_POS_INT + 13 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'9' Help", (15, FONT_POS_INT + 14 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
        cv2.putText(img, "'0' Mute", (15, FONT_POS_INT + 15 * FONT_SPACE_TXT), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
    else:
        pass


    # //=======================================//
    # if  detector_instance.imagepath == "-":
    #     cv2.putText(img, 'FPS: '+str(int(detector_instance.fps)), (20, 40), FONT, FONT_SCALE2, (255,255,255), THICKNESS1)
    # # cv2.imshow("Result5", self.image)

    # cv2.putText(img, f"FPS: {int(detector_instance.fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(img, f"FPS: {detector_instance.fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(img, 'FPS: '+str(int(detector_instance.fps)), (20, 40), FONT, FONT_SCALE2, (255,255,255), THICKNESS1)

    # self.image1 = img.copy()
    # cv2.waitKey(1)
    # time.sleep(5)


    #//=====================================
    # print('objid :=', detector_instance.objid)
    # print('obj :=', detector_instance.obj)
    # print('bbox :=', detector_instance.bbox)
    # print('confidence :=', detector_instance.confidence)
    # print(detector_instance.bbox[0])
    # print(detector_instance.bbox[1])

    # #//=====================================
    # keys = ["objid", "obj", "objbox", "objconf"]
    # values = [detector_instance.objid, detector_instance.obj, detector_instance.bbox, detector_instance.confidence]
    # dict_obj = dict(zip(keys, values))
    # print(dict_obj)

    return img






#//==================================//
class TrafficSignDetector:
    """
    A class for handling object detection and related operations.
    """

    def __init__(self, imagepath, videopath, configpath, modelpath, classespath):
        """
        Initialize the detector class.
        """
        self.imagepath = imagepath
        self.videopath = videopath
        self.configpath = configpath
        self.modelpath = modelpath
        self.classespath = classespath
        self.tracker = Sort()  # Initialize SORT tracker

        print('=== detector_init ===')
        self.setupClasses()  # Set up the detection model
        self.readClasses()  # Read the class labels
        # self.get_gps_location()

    def setupClasses(self):
        """
        Set up the object detection model with the specified configurations.
        """
        print('=== setupClasses ===')
        # Load the pre-trained model and configuration file
        # self.net = cv2.dnn_DetectionModel(modelpath, configpath)
        self.net = cv2.dnn_DetectionModel(modelpath, configpath)

        # Set the model parameters
        self.net.setInputSize(320,326)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def readClasses(self):
        """
        Read the class labels from the specified file and generate random colors for each class.
        """
        print('=== readClasses ===')
        with open(classespath, 'r') as f:
            self.classeslist = f.read().splitlines()

        # Add a background class and generate random colors for each class
        self.classeslist.insert(0, '__Background__')  # Add a placeholder for the background class

        # List of class names to remove
        self.classlabelids_idx = [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]  # Indices of classes to remove
        self.classeslist = [cls for i, cls in enumerate(self.classeslist) if i not in self.classlabelids_idx]
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classeslist), 3))  # Generate random colors for each class

        # All class labels of 92 classes in COCO dataset
        # print(self.classeslist)
        # print(len(self.classeslist))
        # print(self.colorlist)
        # print(len(self.colorlist))

    def onVideo(self):
        """
        Process video input for object detection and display results in real-time.
        """
        from src.config.configs import s  # Import video source configuration

        # Add parent directory to the system path for module imports
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from image_processing.image_process import image_processing
        from timer_processing.timer import Timer

        print('=== onVideo ===')

        # Initialize timers for assistant and alert functionalities
        self.timer_assistant = Timer(sec, command_assistant, self)
        self.timer_alert = Timer(sec, command_alert, self)

        # Check if a video source is provided via command-line arguments
        if len(sys.argv) > 1:
            s = sys.argv[1]  # Use the provided video source
            print(s)

        # Uncomment the following lines to check available camera indices
        # for i in range(10):
        #     cap = cv2.VideoCapture(i)
        #     if cap.isOpened():
        #         print(f"Camera index {i} is available")

        # Open the video source (camera or video file)
        if s == '-':
            print("No video source")
        else:
            self.source = cv2.VideoCapture(s)

        # Set up the display window
        self.window = 'camera'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # Initialize variables for FPS calculation and frame processing
        starttime = 0
        self.id1 = 0
        self.frame_count = 1
        self.obj = str('')
        self.text = str('')
        self.FLAG_MENU = False   # Flag for menu options
        self.volume = 50  # Initial volume level

        # Process video frames in a loop
        while cv2.waitKey(1) != (27 or ord("p")):  # Exit on 'Esc' or 'p' key press
            success, frame = self.source.read()  # Read a frame from the video source

            if not success:
                print("Error: Failed to open video")
                break

            # Calculate FPS
            currenttime = time.time()
            self.fps = 1/(currenttime - starttime)
            starttime = currenttime

            # Get the width and height of the frame
            self.height, self.width, _ = frame.shape
            # print('height:', self.height, 'width:', self.width)

            # Store the original frame and resize it
            self.image_original = frame.copy()
            image_processor = image_processing(self.image_original)
            self.image_resize = image_processor.image_resize(100)

            # //================================//
            # // fps,cam = 32 // fps,vid = 50 //
            # // fps,cam = 12 // fps,vid = 16 //
            # # //==============================//


            # Perform object detection and display results
            bbox, bboxidx, confidence, classlabelids = detect_objects(self, self.net, self.image_resize)
            tracks, track_id_to_class = sort_objects(self, bbox, bboxidx, confidence, classlabelids)
            self.image_out = display_objects(self, self.image_resize, tracks, track_id_to_class)


            # Display FPS on the output image
            cv2.putText(self.image_out, f"FPS: {int(self.fps)}", (15, 80), FONT_FPS, FONT_SIZE_FPS, FONT_COLOR_FPS, FONT_THICKNESS_FPS)

            # Show the processed frame in the display window
            screen_width = 1920
            screen_height = 1080

            # check if the screen resolution is larger than 1920x1080
            if screen_width > 1920 or screen_height > 1080:
                # if screen resolution is larger than 1920x1080, then resize the window to half of the screen resolution
                win_width = int(self.width/2 - int(screen_width/2))
                win_height = int(self.height/2 - int(screen_height/2))
            else:
                win_width = int(self.width/2)
                win_height = int(self.height/2)

            # winx = int(self.width/2 - int(screen_width/2))
            # winy = int(self.height/2 - int(screen_height/2))
            # cv2.moveWindow(self.window, winx, winy)
            cv2.resizeWindow(self.window, win_width, win_height)
            cv2.imshow(self.window, self.image_out)

            # Handle key events for additional functionality
            key = cv2.waitKey(1)
            if key == ord("Q") or key == ord("q") or key == 27:   # Exit on 'Q', 'q', or 'Esc' key press
                # break
                exit_program(self)
            elif key == ord("C") or key == ord("c"):
                print("//=== Press 'C' ===")
                # window = 1
            elif key == ord("B") or key == ord("b"):
                print("//=== Press 'B' ===")
                # break
            elif key == ord("F") or key == ord("f"):
                print("//=== Press 'F' ===")
                # window = 3
            elif key == ord("P") or key == ord("p"):
                print("//=== Press 'P' ===")
                break
            elif key == ord("M") or key == ord("m"):
                print("//=== Press 'M' ===")
                # display_menu(self)
            elif key == ord("S") or key == ord("s") or key == ord("2"):
                print("//=== Press 'S' ===")
                capture_image_objbox(self)
                # cv2.imwrite("image.jpg", self.image_out)
                # cv2.imwrite("image.jpg", self.image_original)
            elif key == ord("+"):
                print("//=== Press '+' ===")
                # increase_sound(self)
            elif key == ord("-"):
                print("//=== Press '-' ===")
                # decrease_sound(self)
            elif key == ord("0"):
                print("//=== Press '0' ===")
                # toggle_sound(self)

            # cv2.putText(img, "'S': Start", (15, 50), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'M': Menu", (15, 70), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'Esc': Exit", (15, 90), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'+' +Sound", (15, 110), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'-' -Sound", (15, 130), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'1': Capture Image", (15, 150), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'2': Capture Image with Boxes", (15, 170), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'3' Pause", (15, 190), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'4' Resume", (15, 210), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'5' Save", (15, 230), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'6' Record", (15, 250), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'7' Play", (15, 270), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'8' Program", (15, 290), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'9' Help", (15, 310), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # cv2.putText(img, "'0' Mute", (15, 330), FONT_TXT, FONT_SIZE_TXT, FONT_COLOR_TXT, FONT_THICKNESS_TXT)
            # End of the video processing loop

        # Stop the timers and release resources
        self.timer_assistant.stop()  # Stop the assistant timer
        self.timer_alert.stop()  # Stop the alert timer
        self.source.release()  # Release the video source

    #//==================================//
    def onImage(self):
        """
        Process a single image for object detection and display results.
        """
        # Add parent directory to the system path for module imports
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from image_processing.image_process import image_processing
        # from timer_processing.timer import timer_record

        print('=== onImage ===')
        # Set up the display window for the image
        self.window = 'image'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # Initialize variables for object detection
        self.id1 = 0  # Object ID counter
        self.obj = str('')  # Detected object labels
        self.text = str('')  # Additional text information
        self.FLAG_MENU = False   # Flag for menu options
        self.volume = 50  # Initial volume level

        self.frame_count = 1  # Frame counter
        # print('imagepath := ', self.imagepath)  # Debug: Print the image path

        # Read the input image from the specified path
        self.image_original = cv2.imread(self.imagepath)
        self.width = self.image_original.shape[1]
        self.height = self.image_original.shape[0]

        # Resize the image using the image processing module
        image_processor = image_processing(self.image_original)
        self.image_resize = image_processor.image_resize(100)

        # Perform object detection and display results
        bbox, bboxidx, confidence, classlabelids = detect_objects(self, self.net, self.image_resize)
        tracks, track_id_to_class = sort_objects(self, bbox, bboxidx, confidence, classlabelids)
        self.image_out = display_objects(self, self.image_resize, tracks, track_id_to_class)

        # Placeholder for additional processing (e.g., text detection)
        self.image_out1 = self.image_out
        self.image_out2 = self.image_out

        # Blend the two processed images (if applicable)
        alpha = 0.5  # Weight for the first image
        beta = (1.0 - alpha)  # Weight for the second image
        self.image_out = cv2.addWeighted(self.image_out1, alpha, self.image_out2, beta, 0.5)
        self.image_out = np.uint8(alpha * (self.image_out1) + beta * (self.image_out2))

        # Show the processed frame in the display window
        screen_width = 1920
        screen_height = 1080

        # check if the screen resolution is larger than 1920x1080
        if screen_width > 1920 or screen_height > 1080:
            # if screen resolution is larger than 1920x1080, then resize the window to half of the screen resolution
            win_width = int(self.width/2 - int(screen_width/2))
            win_height = int(self.height/2 - int(screen_height/2))
        else:
            win_width = int(self.width/2)
            win_height = int(self.height/2)

        # winx = int(self.width/2 - int(screen_width/2))
        # winy = int(self.height/2 - int(screen_height/2))
        # cv2.moveWindow(self.window, winx, winy)
        cv2.resizeWindow(self.window, win_width, win_height)
        # cv2.moveWindow(self.window, winx - 50, winy)

        # Ensure self.image_out is a valid NumPy array
        if not isinstance(self.image_out, np.ndarray):
            self.image_out = np.array(self.image_out, dtype=np.uint8)

        # Ensure the image has valid dimensions
        if len(self.image_out.shape) not in [2, 3]:
            raise ValueError("Invalid image dimensions for cv2.imshow")

        # Display the processed image in the window
        cv2.imshow(self.window, self.image_out)
        cv2.waitKey()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        os._exit(1)  # Exit the program

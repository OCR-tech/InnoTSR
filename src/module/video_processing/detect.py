import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
from voice_alert import voice_alert

class TrafficSignDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ["Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70", 
                            "Speed Limit 80", "End of Speed Limit 80", "Speed Limit 100", "Speed Limit 120", "No Passing",
                            "No Passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", 
                            "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", 
                            "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
                            "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", 
                            "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", 
                            "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", 
                            "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", 
                            "Keep right", "Keep left", "Roundabout mandatory", "End of no passing", 
                            "End of no passing by vehicles over 3.5 metric tons"]


    def detect(self, image):
        image_resized = cv2.resize(image, (64, 64))
        image_array = np.expand_dims(image_resized, axis=0)
        predictions = self.model.predict(image_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]
        return self.class_names[class_index], confidence

    def draw_bounding_box(self, image, box, label, confidence):
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    detector = TrafficSignDetector("models/traffic_sign_model.h5")
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # print("ret",ret)
        # print("frame",frame.shape)
        # print("Processing frame...")
            
        # Uncomment the following lines to convert the frame to different color spaces
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Uncomment if you want to use grayscale
        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Uncomment if you want to use HSV
        # frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Uncomment if you want to use LAB
        # frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)  # Uncomment if you want to use YUV
        # frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)  # Uncomment if you want to use HLS
        # frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  # Uncomment if you want to use YCrCb
        # frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Uncomment if you want to use LAB



        # Resize frame to speed up processing
        frame = cv2.resize(frame,(320,240))                                     # Resize frame to speed up processing
        sign, confidence = detector.detect(frame)
        # print("frame1",frame.shape)



        if confidence > 0.8:
            h, w, _ = frame.shape

            box = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))      # Placeholder for actual bounding box
            # box = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))      # Placeholder for actual bounding box

            detector.draw_bounding_box(frame, box, sign, confidence)
            # voice_alert(f"Traffic sign detected: {sign}")
            voice_alert(f"{sign}")

        cv2.imshow("Traffic Sign Detection", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        key = cv2.waitKey(1)
        if key == ord("Q") or key == ord("q") or key == 27:
            break
        # elif key == ord("C") or key == ord("c"):
        #     image_filter = CANNY


    cap.release()
    cv2.destroyAllWindows()
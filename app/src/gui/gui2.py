'''
# This code creates a simple GUI for a traffic sign recognition system using Tkinter.
# The GUI is a easy-to-use and user-friendly interface for the traffic sign recognition system.
# The GUI includes a label, a canvas for displaying images, and a button to detect traffic signs.
# The GUI is initialized and run in the main block.
# It shows a video stream of the traffic sign detection from the camera in real-time.
# The detect_sign method simulates the detection of a traffic sign and displays the result in a message box.
# The voice_alert method uses the pyttsx3 library to provide a voice alert for the detected traffic sign.
# It allows users to interact with the system and receive feedback on detected traffic signs.
'''



import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading

class TrafficSignGUI:
    def __init__(self, root, detector, speak_func):
        self.root = root
        self.root.title("InnoTSR - Traffic Sign Recognition")
        self.detector = detector
        self.speak = speak_func

        # Video frame
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        # Detect button
        self.detect_btn = tk.Button(root, text="Detect Traffic Sign", command=self.detect_sign)
        self.detect_btn.pack(pady=5)

        # Exit button
        self.exit_btn = tk.Button(root, text="Exit", command=self.on_exit)
        self.exit_btn.pack(pady=5)

        # self.cap = cv2.VideoCapture(0)      # Use 0 for the default camera
        self.cap = cv2.VideoCapture(1)      # Use 1 for the external camera
        self.running = True
        self.update_video()

    def update_video(self):
        # print("=== update_video ===")
        if not self.running:
            return
        ret, frame = self.cap.read()
        # print("ret = ", ret)
        # print("frame.shape = ", frame.shape)

        if ret:
            # Convert the image to RGB and resize for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(20, self.update_video)      # Update every 20 ms

    def detect_sign(self):
        print("=== detect_sign ===")
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            return
        # Run detection (assuming detector has a detect method)
        result = self.detector.detect(frame)
        if result:
            sign_name = result.get("name", "Unknown Sign")
            messagebox.showinfo("Detection Result", f"Detected: {sign_name}")
            if self.speak:
                threading.Thread(target=self.speak, args=(f"Detected {sign_name}",), daemon=True).start()
        else:
            messagebox.showinfo("Detection Result", "No traffic sign detected.")

    def on_exit(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

# Example usage (replace with your actual detector and speak function)
if __name__ == "__main__":

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    from src.video_processing.object_detection import TrafficSignDetector
    from src.voice_processing.voice_output import speak
    from src.config.configs import imagepath, videopath, configpath, modelpath, classespath

    detector = TrafficSignDetector(imagepath, videopath, configpath, modelpath, classespath)

    root = tk.Tk()
    app = TrafficSignGUI(root, detector, speak)
    root.mainloop()





# Import necessary libraries
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk



# Define the main application class
class TrafficSignApp:
    """
    A simple GUI application for traffic sign recognition using Tkinter.
    """
    def __init__(self, root):
        """
        Initializes the TrafficSignApp with a Tkinter root window.
        """

        # Initialize the Tkinter root window
        self.root = root
        self.root.title("Traffic Sign Recognition System")

        # Set the window size
        self.label = tk.Label(root, text="Traffic Sign Recognition System", font=("Helvetica", 16))
        self.label.pack(pady=20)

        # Create a canvas to display images
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_detect = tk.Button(root, text="Detect Traffic Sign", command=self.detect_sign)
        self.btn_detect.pack(pady=20)


    def detect_sign(self):
        """
        Simulate traffic sign detection and display the result.
        """
        # Simulate detection of a traffic sign (e.g., stop sign)
        sign = "Stop Sign"
        messagebox.showinfo("Traffic Sign Detected", f"Detected: {sign}")
        # self.voice_alert(f"Traffic sign detected: {sign}")
        self.voice_alert(f"{sign}")

    def voice_alert(self, message):
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()



if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()

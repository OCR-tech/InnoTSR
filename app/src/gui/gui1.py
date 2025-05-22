import tkinter as tk
from PIL import Image, ImageTk
import cv2

class SimpleCam:
    def __init__(self, root):
        self.root = root
        self.label = tk.Label(root)
        self.label.pack()
        self.cap = cv2.VideoCapture(1)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.root.after(20, self.update)

root = tk.Tk()
app = SimpleCam(root)
root.mainloop()

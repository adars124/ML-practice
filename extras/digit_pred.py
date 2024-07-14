import tkinter as tk
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

# Define interface parameters
WIDTH = 280
HEIGHT = 280

# Loading the saved model
model = joblib.load('random_forest_model.pkl')

class App(tk.Tk):
    
    def __init__(self):
        super().__init__()

        self.conf = 0.0

        self.title("Digit Prediction using Random Forest Classification")
        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack() # Add the canvas to the original frame

        self.label = tk.Label(self, text="", font=("Helvetica", 24))
        self.label.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)

        self.prob_label = tk.Label(self, text=f"Probability: {self.conf}", font=("Helvetica", 12))
        self.prob_label.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=50)

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.predict)
        self.bind('n', self.clear_canvas)

        self.image1 = Image.new("L", (WIDTH, HEIGHT), (255))
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        x1, y1 = (event.x - 12), (event.y - 12)
        x2, y2 = (event.x + 12), (event.y + 12)

        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self, event=None):
        self.canvas.delete('all')
        self.image1 = Image.new('L', (WIDTH, HEIGHT), (255))
        self.draw = ImageDraw.Draw(self.image1)
        self.label.config(text='')
        self.prob_label.config(text=f"Probability: {self.conf}")

    def predict(self, event=None):
        img = self.image1.resize((28, 28))
        img = ImageOps.invert(img)
        
        # Display the resized image for verification
        # plt.imshow(img, cmap='gray')
        # plt.title("Resized Image")
        # plt.show()

        img = np.array(img).reshape(1, -1)
        
        pred = model.predict(img)[0]
        probs = model.predict_proba(img)[0]

        confidence = max(probs) * 100
        self.label.config(text=str(pred))
        self.prob_label.config(text=f"Probability: {confidence:.2f}")

if __name__ == "__main__":
    app = App()
    app.mainloop()

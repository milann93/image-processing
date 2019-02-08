import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


os.chdir('/home/milan/Desktop/image-processing')

# Need GUI for selecting pictures
img = Image.open('warrior.jpeg')
# greyscale = img.convert('L')
# hist = histogram(grayscale, 256)
# hist = np.array(hist)
# hist = np.reshape(hist, (256, 1))
# histImg = Image.fromarray(hist, 'L')

window = tk.Tk()
window.geometry('700x700')
rot = tk.DoubleVar()


def refresh():
    global img # global reference to img

    angle = rot.get()
    imgCopy = img.rotate(angle, expand=True)

    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo

    rotText = 'Value = ' + str(angle)
    lblRotVal.config(text=rotText)


# ROTATION
lblRot = tk.Label(window, text='Rotation')
lblRot.grid(column=0, row=0)

sliderRot = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL, variable=rot, resolution=0.5)
sliderRot.grid(column=1, row=0)


lblRotVal = tk.Label(window, text='Value = 0')
lblRotVal.grid(column=2, row=0)




refreshButton = tk.Button(window, text='Refresh', command=refresh)
refreshButton.grid(column=1, row=20) # row as placeholder

# Displaying image
photo = ImageTk.PhotoImage(img)
photoLabel = tk.Label(image=photo)
photoLabel.grid(column=4, row=21) # row as placeholder

# Displaying histogram of image
# histogram = ImageTk.PhotoImage(histImg)
# histogramLabel = tk.Label(image=histogram)
# histogramLabel.grid(column=5,row=21) # row as placeholder

window.mainloop()

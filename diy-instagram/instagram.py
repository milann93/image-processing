import os
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np

import zoom
import rgb
import hsv


# os.chdir('/home/milan/Desktop/image-processing')

img = Image.open('./data/warrior.jpeg')
# img = Image.open('lenacolor.tif')
w, h = img.size

window = tk.Tk()
window.title('Instagram')
window.geometry('920x1080')

r = tk.IntVar()
g = tk.IntVar()
b = tk.IntVar()
rot = tk.IntVar()
brightness = tk.IntVar()
warmth = tk.IntVar()
saturation = tk.IntVar()
contrast = tk.IntVar()
fade = tk.IntVar()
highlights = tk.IntVar()
shadow = tk.IntVar()
vignet = tk.IntVar()
sharpen = tk.IntVar()
tilt = tk.IntVar()
radio = tk.StringVar()
zm = tk.IntVar()
re = tk.IntVar()


def apply():
    global img # global reference to img
    imgCopy = img.copy()

    # Applying ZOOM
    zoomCoef = zm.get()
    imgCopy = zoom.zoom(imgCopy, zoomCoef)

    # Applying RESCALE
    rescaleCoef = re.get()
    imgCopy = zoom.rescaling(imgCopy, rescaleCoef)


    # Applying RBG color change
    r1 = r.get()
    g1 = g.get()
    b1 = b.get()


    warmCoef = warmth.get()
    fadeCoef = fade.get()
    rgb.rgbChange(imgCopy, r1, g1, b1, warmCoef, fadeCoef)

    # Changing BRIGHTNESS, SATURATION, CONTRAST
    vCoef = brightness.get()
    sCoef = saturation.get()
    contrastBoundary= contrast.get()
    imgCopy = hsv.rgb2hsv(imgCopy, sCoef, vCoef, contrastBoundary)

    # Changing SHARPNESS
    sharpCoef = sharpen.get()
    imgCopy = hsv.adjustSharpness(imgCopy, sharpCoef)

    # Applying HIGHLIGHTS
    hCoef = highlights.get()
    imgCopy = hsv.adjustHighlights(imgCopy, hCoef)

    # Applying SHADOW
    shCoef = shadow.get()
    imgCopy = hsv.adjustShadow(imgCopy, shCoef)

    # Applying VIGNETTE
    vinCoef = vignet.get()
    imgCopy = hsv.vignette(imgCopy, vinCoef)

    # Applying TILT SHIFT
    tiltcoef = tilt.get()
    radCeof = radio.get()
    imgCopy = hsv.tiltShift(imgCopy, tiltcoef, radCeof)

    hist = hsv.histogram(imgCopy)

    # Changing ANGLE
    angle = rot.get()
    imgCopy = imgCopy.convert('RGB')
    imgCopy = rgb.rotate(imgCopy, angle * np.pi / 180)

    # Displaying changed image
    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo

    histogr = ImageTk.PhotoImage(hist)
    histLabel = tk.Label(image=histogr)
    histLabel.image = histogr
    histLabel.place(x=500, y=h+50)


def default():
    global img
    imgCopy = img.copy()

    imgCopy = zoom.zoom(imgCopy, 0)
    sliderZoom.set(0)

    imgCopy = zoom.rescaling(imgCopy, 0)
    sliderRescale.set(0)

    rgb.rgbChange(imgCopy, 0, 0, 0, 0, 0)
    sliderR.set(0)
    sliderG.set(0)
    sliderB.set(0)
    sliderWarmth.set(0)
    sliderFade.set(0)

    imgCopy = rgb.rotate(imgCopy, 0)
    sliderRot.set(0)

    imgCopy = hsv.rgb2hsv(imgCopy, 0, 0, 0)
    sliderBrightness.set(0)
    sliderSaturation.set(0)
    sliderContrast.set(0)

    imgCopy = hsv.adjustSharpness(imgCopy, 0)
    sliderSharpen.set(0)

    imgCopy = hsv.adjustHighlights(imgCopy, 0)
    sliderHighlights.set(0)

    imgCopy = hsv.adjustShadow(imgCopy, 0)
    sliderShadow.set(0)

    imgCopy = hsv.vignette(imgCopy, 0)
    sliderVignette.set(0)

    imgCopy = hsv.tiltShift(imgCopy, 0, 0)
    sliderTiltShift.set(0)

    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo

    hist = hsv.histogram(imgCopy)

    histogr = ImageTk.PhotoImage(hist)
    histLabel = tk.Label(image=histogr)
    histLabel.image = histogr
    histLabel.place(x=500, y=h+50)


###############
# GUI elements
###############

# RGB
sliderR = tk.Scale(window, from_=100, to=-100, variable=r) # range goes from higher to lower because of scale orientation
sliderR.grid(column=0, row=0)
lblR = tk.Label(window, text='    R')  # additional spaces for formating
lblR.grid(column=0, row=1)

sliderG = tk.Scale(window, from_=100, to=-100, variable=g)
sliderG.grid(column=1, row=0)
lblG = tk.Label(window, text='    G')
lblG.grid(column=1, row=1)

sliderB = tk.Scale(window, from_=100, to=-100, variable=b)
sliderB.grid(column=2, row=0)
lblB = tk.Label(window, text='    B')
lblB.grid(column=2, row=1)

# ROTATION
lblRot = tk.Label(window, text='Rotation')
lblRot.grid(column=0, row=2)

sliderRot = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL, variable=rot)
sliderRot.grid(column=1, row=2)


# BRIGHTNESS
lblBrightness = tk.Label(window, text='Brightness')
lblBrightness.grid(column=0, row=3)

sliderBrightness = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=brightness)
sliderBrightness.grid(column=1, row=3)


# CONTRAST
lblContrast = tk.Label(window, text='Contrast')
lblContrast.grid(column=0, row=4)

sliderContrast = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=contrast)
sliderContrast.grid(column=1, row=4)

# WARMTH
lblWarmth = tk.Label(window, text='Warmth')
lblWarmth.grid(column=0, row=5)

sliderWarmth = tk.Scale(window, from_=-100, to =100, orient=tk.HORIZONTAL, variable=warmth)
sliderWarmth.grid(column=1, row=5)

# SATURATION
lblSaturation = tk.Label(window, text='Saturation')
lblSaturation.grid(column=0, row=6)

sliderSaturation = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=saturation)
sliderSaturation.grid(column=1, row=6)

# FADE
lblFade = tk.Label(window, text='Fade')
lblFade.grid(column=0, row=7)

sliderFade = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=fade)
sliderFade.grid(column=1, row=7)

# HIGHLIGHTS
lblHighlights = tk.Label(window, text='Highlights')
lblHighlights.grid(column=0, row=8)

sliderHighlights = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=highlights)
sliderHighlights.grid(column=1, row=8)

# SHADOWS
lblShadow = tk.Label(window, text='Shadow')
lblShadow.grid(column=0, row=9)

sliderShadow = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=shadow)
sliderShadow.grid(column=1, row=9)

# VIGNETTE
lblVignette = tk.Label(window, text='Vignette')
lblVignette.grid(column=0, row=10)

sliderVignette = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=vignet)
sliderVignette.grid(column=1, row=10)

# TILT SHIFT
lblTiltShift = tk.Label(window, text='Tilt shift')
lblTiltShift.grid(column=0, row = 11)

sliderTiltShift = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=tilt)
sliderTiltShift.grid(column=1, row=11)

radioLinear = tk.Radiobutton(window, text='Linear', variable=radio, value='linear')
radioLinear.grid(column=3, row=11)

radioRadial = tk.Radiobutton(window, text='Radial', variable=radio, value='radial')
radioRadial.grid(column=4, row=11)

# SHARPEN
lblSharpen = tk.Label(window, text='Sharpen')
lblSharpen.grid(column=0, row=12)

sliderSharpen = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=sharpen)
sliderSharpen.grid(column=1, row=12)


# ZOOM
lblZoom = tk.Label(window, text='Zoom')
lblZoom.grid(column=0, row=13)

sliderZoom = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=zm)
sliderZoom.grid(column=1, row=13)


# RESCALE
lblRescale = tk.Label(window, text='Rescale')
lblRescale.grid(column=0, row=14)

sliderRescale = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=re)
sliderRescale.grid(column=1, row=14)

# BUTTONS
applyButton = tk.Button(window, text='Apply', command=apply)
applyButton.grid(column=1, row=20) # row as placeholder

defaultButton = tk.Button(window, text='Default', command=default)
defaultButton.grid(column=1, row=21)

# Displaying image
photo = ImageTk.PhotoImage(img)
photoLabel = tk.Label(image=photo)
photoLabel.image = photo
# photoLabel.grid(column=5, row=21) # row as placeholder
photoLabel.place(x=500, y=0)

window.mainloop()

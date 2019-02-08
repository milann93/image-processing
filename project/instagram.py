import os
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from scipy import ndimage

os.chdir('/home/milan/Desktop/image-processing')

# Need GUI for selecting pictures
img = Image.open('warrior.jpeg')
# img = Image.open('test.jpg')
# greyscale = img.convert('L')
# hist = histogram(grayscale, 256)
# hist = np.array(hist)
# hist = np.reshape(hist, (256, 1))
# histImg = Image.fromarray(hist, 'L')

window = tk.Tk()
window.title('Instagram')
window.geometry('700x700')

r = tk.IntVar()
g = tk.IntVar()
b = tk.IntVar()
rot = tk.IntVar()
brightness = tk.IntVar()
warmth = tk.IntVar()
saturation = tk.IntVar()
contrast = tk.IntVar()
fade = tk.IntVar()
sharpen = tk.IntVar()


def limit(c): # this will return pixel value to the range (0 - 255)
    if c >= 255:
        return 255
    elif c <= 0:
        return 0
    else:
        return int(c)


def adjustWarmth(r, g, b, warmCoef):
    if warmCoef >= 0:
        change = 0.5 * warmCoef/100 + 1
        r *= change
        g *= change
        b /= change
    else:
        change = 0.5 * (warmCoef+100)/100 + 0.5
        r *= change
        g /= change
        b /= change

    text = 'Warmth = {0}'.format(warmCoef)
    lblWarmthVal.config(text=text)
    return r, g, b


def adjustFade(r, g, b, fadeCoef):
    # Adds gray image over original
    # fadeCoef is divided by 2 to lower the fading effect

    r += fadeCoef/2
    g += fadeCoef/2
    b += fadeCoef/2

    text = 'Fade = {0}'.format(fadeCoef)
    lblFadeVal.config(text=text)
    return r, g, b


def rgbChange(img, r, g, b, warmCoef, fadeCoef):

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r1, g1, b1 = img.getpixel((i, j))

            # r1 = limit(r1 + r)
            # g1 = limit(g1 + g)
            # b1 = limit(b1 + b)

            r1 += r
            g1 += g
            b1 += b

            r1, g1, b1 = adjustWarmth(r1, g1, b1, warmCoef)

            r1, g1, b1 = adjustFade(r1, g1, b1, fadeCoef)

            r1 = limit(r1)
            g1 = limit(g1)
            b1 = limit(b1)

            img.putpixel((i, j), (r1, g1, b1))


def rotation(img, angle, expand):
    img = img.rotate(angle=angle, expand=expand)
    text = 'Angle = ' + str(angle)
    lblRotVal.config(text=text)
    return img


def adjustEffect(x, coef):
    # Applicable to saturation and brightness
    if coef >= 0:
        coef = 0.5 * coef/100 + 1 # Scales interval [0, 100] to [1, 1.5]
    else:
        coef = 0.5 * (coef + 100)/100 + 0.5 # Scales interval [-100, 0] to [0.5, 1]

    x *= coef
    if x > 255:
        x = 255
    return x


def adjustContrast(v, contrastBoundary=0):
    contrastBoundary = 1.22*contrastBoundary # range of contrast boundary is [0, 122]

    if v <= contrastBoundary:
        v = 0
    elif v >= 255-contrastBoundary:
        v = 255
    elif v < 122:
        v = (v-contrastBoundary)/(122-contrastBoundary) * 122
    else:
        v = (v-123)/(255-contrastBoundary-123) * 122 + 123
    return v


def rgb2hsv(img, sCoef = 1, vCoef = 1, contrastBoundary=0):

    # Takes RGB image as input and converts it to HSV color space
    # hueCoef, sCoef and vCoef are additional coefficients that are used to scale hue, saturation and intensity respectively

    # Note:
    # Hue range is from 0 to 360
    # Saturation and intensity have range from 0 to 100 (percents)

    # PIL Image object takes values from 0 to 255 so ranges for hue, saturation and intensity need to be scaled to fit that range

    w, h = img.size
    hsvimg = Image.new('HSV', (w, h))

    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))

            # v = (r + g + b) / 3

            r = r/255
            g = g/255
            b = b/255

            cmax = max(r, g, b)
            cmin = min(r, g, b)
            df = cmax - cmin

            if 0 == df:
                hue = 0
            elif cmax == r:
                hue = (51 * ((g-b)/df) + 255) % 255
            elif cmax == g:
                hue = (51 * ((b-r)/df) + 85) % 255
            elif cmax == b:
                hue = (51 * ((r-g)/df) + 170) % 255

            if 0 == cmax:
                s = 0
            else:
                s = df / cmax

            # returning to range (0 - 255)
            s = s *255
            v = cmax * 255


            hue = hue

            # Saturation adjustment
            s = adjustEffect(s, sCoef)

            # Brightness adjustment
            v = adjustEffect(v, vCoef)

            # Contrast adjustment
            v = adjustContrast(v, contrastBoundary)

            hue = int(hue)
            s = int(s)
            v = int(v)

            hsvimg.putpixel((i, j), (hue, s, v))

    return hsvimg


def transformationsInHSVspectra(img, sCoef, vCoef, contrastBoundary):
    text = 'Brightness = {0}'.format(vCoef)
    lblBrightnessVal.config(text=text)

    text = 'Saturation = {0}'.format(sCoef)
    lblSaturationVal.config(text=text)

    text = 'Contrast = {0}'.format(contrastBoundary)
    lblContrastVal.config(text=text)

    return rgb2hsv(img, sCoef, vCoef, contrastBoundary)


def adjustSharpness(img, c):
    # Sharpend image using unsharp masking

    # puting c into range [1, 0.6], where c = 1 will give original image
    # sharpness increases as c reduces

    c = -0.4 * (c/100) + 1

    w, h = img.size
    npimg = np.array(img)

    # changing type from uint8 to int to avoid overflow and underflow during subtraction
    npimg = npimg.astype(int)

    mask = 1/49 * np.ones((7, 7), dtype=int)

    npimg[:, :, 0] = (c/(2*c-1)) * npimg[:, :, 0] - ((1-c)/(2*c-1)) * ndimage.convolve(npimg[:, :, 0], mask)
    npimg[:, :, 1] = (c/(2*c-1)) * npimg[:, :, 1] - ((1-c)/(2*c-1)) * ndimage.convolve(npimg[:, :, 1], mask)
    npimg[:, :, 2] = (c/(2*c-1)) * npimg[:, :, 2] - ((1-c)/(2*c-1)) * ndimage.convolve(npimg[:, :, 2], mask)

    # manually returning values to range
    for i in range(w):
        for j in range(h):
            for k in range(3):
                npimg[j, i, k] = limit(npimg[j, i, k])

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'RGB')
    return pilImg


def apply():
    global img # global reference to img
    imgCopy = img.copy()

    # Applying RBG color change
    r1 = r.get()
    g1 = g.get()
    b1 = b.get()

    sharpCoef = sharpen.get()
    imgCopy = adjustSharpness(imgCopy, sharpCoef)

    warmCoef = warmth.get()
    fadeCoef = fade.get()
    rgbChange(imgCopy, r1, g1, b1, warmCoef, fadeCoef)

    # Changing BRIGHTNESS, SATURATION, CONTRAST
    vCoef = brightness.get()
    sCoef = saturation.get()
    contrastBoundary= contrast.get()
    imgCopy = transformationsInHSVspectra(imgCopy, sCoef, vCoef, contrastBoundary)


    # Changing ANGLE
    angle = rot.get()
    imgCopy = rotation(imgCopy, angle, True)


    # Displaying changed image
    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo


def default():
    global img
    imgCopy = img.copy()

    rgbChange(imgCopy, 0, 0, 0, 0, 0)
    sliderR.set(0)
    sliderG.set(0)
    sliderB.set(0)
    sliderWarmth.set(0)
    sliderFade.set(0)

    imgCopy = rotation(imgCopy, 0, True)
    sliderRot.set(0)

    imgCopy = transformationsInHSVspectra(imgCopy, 0, 0, 0)
    sliderBrightness.set(0)
    sliderSaturation.set(0)
    sliderContrast.set(0)

    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo


###############
# GUI elements
###############

# RGB
sliderR = tk.Scale(window, from_=100, to=-100, variable=r) # range goes from higher to lower because of scale orientation
sliderR.grid(column=0, row=0)
lblR = tk.Label(window, text='    R') # additional spaces for formating
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

lblRotVal = tk.Label(window, text='Angle = 0')
lblRotVal.grid(column=2, row=2, columnspan=2)

# BRIGHTNESS
lblBrightness = tk.Label(window, text='Brightness')
lblBrightness.grid(column=0, row=3)

sliderBrightness = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=brightness)
sliderBrightness.grid(column=1, row=3)

lblBrightnessVal = tk.Label(window, text='Brightness = 0')
lblBrightnessVal.grid(column=2, row=3, columnspan=2)

# CONTRAST
lblContrast = tk.Label(window, text='Contrast')
lblContrast.grid(column=0, row=4)

sliderContrast = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=contrast)
sliderContrast.grid(column=1, row=4)

lblContrastVal = tk.Label(window, text='Contrast = 0')
lblContrastVal.grid(column=2, row=4)

# WARMTH
lblWarmth = tk.Label(window, text='Warmth')
lblWarmth.grid(column=0, row=5)

sliderWarmth = tk.Scale(window, from_=-100, to =100, orient=tk.HORIZONTAL, variable=warmth)
sliderWarmth.grid(column=1, row=5)

lblWarmthVal = tk.Label(window, text='Warmth = 0')
lblWarmthVal.grid(column=2, row=5)

# SATURATION
lblSaturation = tk.Label(window, text='Saturation')
lblSaturation.grid(column=0, row=6)

sliderSaturation = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=saturation)
sliderSaturation.grid(column=1, row=6)

lblSaturationVal = tk.Label(window, text='Saturation = 0')
lblSaturationVal.grid(column=2, row=6, columnspan=2)

# FADE
lblFade = tk.Label(window, text='Fade')
lblFade.grid(column=0, row=7)

sliderFade = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=fade)
sliderFade.grid(column=1, row=7)

lblFadeVal = tk.Label(window, text='Fade = 0')
lblFadeVal.grid(column=2, row=7)

# HIGHLIGHTS
# 8

# SHADOWS
# 9

# VIGNETTE
# 10

# TILT SHIFT
# 11

# SHARPEN
lblSharpen = tk.Label(window, text='Sharpen')
lblSharpen.grid(column=0, row=12)

sliderSharpen = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=sharpen)
sliderSharpen.grid(column=1, row=12)

lblSharpenVal = tk.Label(window, text='Sharpen = 0')
lblSharpenVal.grid(column=2, row=12)

# ZOOM


# BUTTONS
applyButton = tk.Button(window, text='Apply', command=apply)
applyButton.grid(column=1, row=20) # row as placeholder

defaultButton = tk.Button(window, text='Default', command=default)
defaultButton.grid(column=1, row=21)

# Displaying image
photo = ImageTk.PhotoImage(img)
photoLabel = tk.Label(image=photo)
photoLabel.image = photo
photoLabel.grid(column=4, row=21) # row as placeholder

# Displaying histogram of image
# histogram = ImageTk.PhotoImage(histImg)
# histogramLabel = tk.Label(image=histogram)
# histogramLabel.grid(column=5,row=21) # row as placeholder

window.mainloop()

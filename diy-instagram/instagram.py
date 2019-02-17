import os
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import math

os.chdir('/home/milan/Desktop/image-processing')

img = Image.open('warrior.jpeg')
# img = Image.open('lenacolor.tif')

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


def limit(c):
    # returns pixel value to the range (0 - 255)
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


def adjustBrightness(x, coef):
    text = 'Brightness = {0}'.format(coef)
    lblBrightnessVal.config(text=text)

    if coef >= 0:
        coef= 0.5 * coef/100 + 1 # Scales interval [0, 100] to [1, 1.5]
    else:
        coef = 0.5 * (coef + 100)/100 + 0.5 # Scales interval [-100, 0] to [0.5, 1]

    x *= coef
    if x > 255:
        x = 255

    return x


def adjustSaturation(x, coef):
    text = 'Saturation = {0}'.format(coef)
    lblSaturationVal.config(text=text)

    if coef >= 0:
        coef= 0.5 * coef/100 + 1 # Scales interval [0, 100] to [1, 1.5]
    else:
        coef = 0.5 * (coef + 100)/100 + 0.5 # Scales interval [-100, 0] to [0.5, 1]

    x *= coef
    if x > 255:
        x = 255

    return x


def adjustContrast(v, contrastBoundary):
    text = 'Contrast = {0}'.format(contrastBoundary)
    lblContrastVal.config(text=text)

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

            v = (r + g + b) / 3

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
            # v = cmax * 255

            # Saturation adjustment
            s = adjustSaturation(s, sCoef)

            # Brightness adjustment
            v = adjustBrightness(v, vCoef)

            # Contrast adjustment
            v = adjustContrast(v, contrastBoundary)

            hue = int(hue)
            s = int(s)
            v = int(v)

            hsvimg.putpixel((i, j), (hue, s, v))

    return hsvimg


def adjustSharpness(hsvimg, c):
    # Sharpend image using unsharp masking

    # puting c into range [1, 0.6], where c = 1 will give original image
    # sharpness increases as c reduces

    text = 'Sharpen = {0}'.format(c)
    lblSharpenVal.config(text=text)

    c = -0.4 * (c/100) + 1

    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    # changing type from uint8 to int to avoid overflow and underflow during subtraction
    npimg = npimg.astype(int)

    mask = 1/49 * np.ones((7, 7), dtype=int)
    blured = ndimage.convolve(npimg[:, :, 2], mask)

    npimg[:, :, 2] = (c/(2*c-1)) * npimg[:, :, 2] - ((1-c)/(2*c-1)) * blured

    # manually returning values to range
    for i in range(w):
        for j in range(h):
                npimg[j, i, 2] = limit(npimg[j, i, 2])

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'HSV')
    return pilImg


def vignette(hsvimg, coef):
    text = 'Vignette = {0}'.format(coef)
    lblVignetteVal.config(text=text)

    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    # changing type from uint8 to int to avoid overflow and underflow during subtraction
    npimg = npimg.astype(int)

    x = np.ones((w, 1))
    y = np.ones((h, 1))

    bound_y = (coef/100) * (1/3) * h
    bound_y = int(bound_y)

    bound_x = (coef/100) * (1/3) * w
    bound_x = int(bound_x)

    for i in range(bound_y):
        y[i] = i / bound_y
        y[h - 1 - i] = y[i]

    for i in range(bound_x):
        x[i] = i / bound_x
        x[w - 1 - i] = x[i]

    mask = y.dot(x.T)

    npimg[:, :, 2] = npimg[:, :, 2] * mask

        # manually returning values to range
    for i in range(w):
        for j in range(h):
                npimg[j, i, 2] = limit(npimg[j, i, 2])

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'HSV')
    return pilImg


def adjustShadow(hsvimg, coef):
    text = 'Shadow = {0}'.format(coef)
    lblShadowVal.config(text=text)

    c = -0.5 * (coef/100) + 1

    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    # changing type from uint8 to float to avoid overflow and underflow during subtraction and to be able to divide
    npimg = npimg.astype(float)

    for i in range(w):
        for j in range(h):
            a = npimg[j, i, 2]
            if npimg[j, i, 2] < 123:
                npimg[j, i, 2] /=123
                npimg[j, i, 2] **= c
                npimg[j, i, 2] *= 123
                npimg[j, i, 2] = np.round(npimg[j, i, 2])

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'HSV')
    return pilImg


def adjustHighlights(hsvimg, coef):
    text = 'Highlights = {0}'.format(coef)
    lblHighlightsVal.config(text=text)

    c = -0.3 * (coef/100) + 1

    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    # changing type from uint8 to float to avoid overflow and underflow during subtraction and to be able to divide
    npimg = npimg.astype(float)

    for i in range(w):
        for j in range(h):
            if npimg[j, i, 2] > 50:
                npimg[j, i, 2] /=255
                npimg[j, i, 2] **= c
                npimg[j, i, 2] *= 255

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'HSV')
    return pilImg


def tiltShift(hsvimg, coef, mode):
    # mode can be linear or radial
    # different masks are used for blurring, depending of mode, for better results

    w, h = hsvimg.size

    npimg = np.array(hsvimg)
    npimg = npimg.astype(int)

    bound = (coef/100) * (1/3) * h
    bound = int(bound)

    if 'linear' == mode:
        mask = 1/9 * np.ones((3, 3), dtype=int)
        npimg[0:bound, :, 2] = ndimage.convolve(npimg[0:bound, :, 2], mask)
        npimg[h-bound:h, :, 2] = ndimage.convolve(npimg[h-bound:h, :, 2], mask)

    if 'radial' == mode:
        mask = 1/25 * np.ones((5, 5), dtype=int)
        binaryMask = np.zeros((h, w))

        maxdf = math.sqrt(math.pow(w/2, 2) + math.pow(h/2, 2)) # largest distance from the middle
        x = max(w, h) / 3 # 1/3 of larger dim

        # Scaling to range [maxdf, x]
        coef = (x - maxdf) * (coef/100) + maxdf

        for i in range(w):
            for j in range(h):
                # df = np.abs(w/2 - i) + np.abs(h/2 - j)
                df = math.sqrt(math.pow(int(w/2) - i, 2) + math.pow(int(h/2) - j, 2))
                if df > coef:
                    binaryMask[j, i] = 1

        # filtered = npimg[:, :, 2] * binaryMask
        blured = ndimage.convolve(npimg[:, :, 2], mask)

        for i in range(w):
            for j in range(h):
                if 1 == binaryMask[j, i]:
                    npimg[j, i, 2] = blured[j, i]



    # manually returning values to range
    for i in range(w):
        for j in range(h):
            npimg[j, i, 2] = limit(npimg[j, i, 2])

    # returning tensor to uint8, so it can be properly converted into PIL image
    npimg = npimg.astype(np.uint8)
    pilImg = Image.fromarray(npimg, 'HSV')
    return pilImg


def rotate_coords(x, y, theta, ox, oy):
    # Rotate arrays of coordinates x and y by theta radians about the
    # point (ox, oy).

    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy

def rotate_image(img, theta, fill=255):
    # Rotate the image src by theta radians about (ox, oy).
    # Pixels in the result that don't correspond to pixels in src are
    # replaced by the value fill.

    # Images have origin at the top left, so negate the angle.
    theta = -theta

    sh, sw = img.shape
    # Finding center of the image
    ox = int(sh/2)
    oy = int(sw/2)

    # Rotated positions of the corners of the source image.
    cx, cy = rotate_coords([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)

    # Determine dimensions of destination image.
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

    # Coordinates of pixels in destination image.
    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))

    # Corresponding coordinates in source image. Since we are
    # transforming dest-to-src here, the rotation is negated.
    sx, sy = rotate_coords(dx + cx.min(), dy + cy.min(), -theta, ox, oy)

    # Select nearest neighbour.
    sx, sy = sx.round().astype(int), sy.round().astype(int)

    # Mask for valid coordinates.
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)

    # Create destination image.
    dest = np.empty(shape=(dh, dw), dtype=img.dtype)

    # Copy valid coordinates from source image.
    dest[dy[mask], dx[mask]] = img[sy[mask], sx[mask]]

    # Fill invalid coordinates.
    dest[dy[~mask], dx[~mask]] = fill

    return dest


def rotate(img, theta):
    npimg = np.array(img)

    R = rotate_image(npimg[:, :, 0], theta)
    G = rotate_image(npimg[:, :, 1], theta)
    B = rotate_image(npimg[:, :, 2], theta)

    h, w = R.shape
    rgbimg = np.zeros((h, w, 3), dtype=np.uint8)
    print(rgbimg.shape)
    for j in range(w):
        for i in range(h):
            rgbimg[i, j] = np.array([R[i, j], G[i, j], B[i, j]])

    rgbimg = Image.fromarray(rgbimg, 'RGB')
    return rgbimg

def histogram(hsvimg):
    n = np.arange(0, 256)
    hist = np.zeros(256)
    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    for i in range(w):
        for j in range(h):
            hist[npimg[:, :, 2][j, i]] += 1

    # fig = plt.figure()
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.bar(n, hist)

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pilImg = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return pilImg

def apply():
    global img # global reference to img
    imgCopy = img.copy()

    # Applying RBG color change
    r1 = r.get()
    g1 = g.get()
    b1 = b.get()


    warmCoef = warmth.get()
    fadeCoef = fade.get()
    rgbChange(imgCopy, r1, g1, b1, warmCoef, fadeCoef)

    # Changing BRIGHTNESS, SATURATION, CONTRAST
    vCoef = brightness.get()
    sCoef = saturation.get()
    contrastBoundary= contrast.get()
    imgCopy = rgb2hsv(imgCopy, sCoef, vCoef, contrastBoundary)

    # Changing SHARPNESS
    sharpCoef = sharpen.get()
    imgCopy = adjustSharpness(imgCopy, sharpCoef)

    # Applying HIGHLIGHTS
    hCoef = highlights.get()
    imgCopy = adjustHighlights(imgCopy, hCoef)

    # Applying SHADOW
    shCoef = shadow.get()
    imgCopy = adjustShadow(imgCopy, shCoef)

    # Applying VIGNETTE
    vinCoef = vignet.get()
    imgCopy = vignette(imgCopy, vinCoef)

    # Applying TILT SHIFT
    tiltcoef = tilt.get()
    radCeof = radio.get()
    imgCopy = tiltShift(imgCopy, tiltcoef, radCeof)

    hist = histogram(imgCopy)

    # Changing ANGLE
    angle = rot.get()
    imgCopy = imgCopy.convert('RGB')
    imgCopy = rotate(imgCopy, angle * np.pi / 180)

    # Displaying changed image
    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo

    histogr = ImageTk.PhotoImage(hist)
    histLabel = tk.Label(image=histogr)
    histLabel.image = histogr
    histLabel.grid(column=5, row=22)



def default():
    global img
    imgCopy = img.copy()

    rgbChange(imgCopy, 0, 0, 0, 0, 0)
    sliderR.set(0)
    sliderG.set(0)
    sliderB.set(0)
    sliderWarmth.set(0)
    sliderFade.set(0)

    imgCopy = rotate(imgCopy, 0)
    sliderRot.set(0)

    imgCopy = rgb2hsv(imgCopy, 0, 0, 0)
    sliderBrightness.set(0)
    sliderSaturation.set(0)
    sliderContrast.set(0)

    imgCopy = adjustSharpness(imgCopy, 0)
    sliderSharpen.set(0)

    imgCopy = adjustHighlights(imgCopy, 0)
    sliderHighlights.set(0)

    imgCopy = adjustShadow(imgCopy, 0)
    sliderShadow.set(0)

    imgCopy = vignette(imgCopy, 0)
    sliderVignette.set(0)

    imgCopy = tiltShift(imgCopy, 0, 0)
    sliderTiltShift.set(0)

    photo = ImageTk.PhotoImage(imgCopy)
    photoLabel.config(image=photo)
    photoLabel.image = photo

    hist = histogram(imgCopy)

    histogr = ImageTk.PhotoImage(hist)
    histLabel = tk.Label(image=histogr)
    histLabel.image = histogr
    histLabel.grid(column=5, row=22)


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
lblHighlights = tk.Label(window, text='Highlights')
lblHighlights.grid(column=0, row=8)

sliderHighlights = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=highlights)
sliderHighlights.grid(column=1, row=8)

lblHighlightsVal = tk.Label(window, text='Highlights = 0')
lblHighlightsVal.grid(column=2, row=8)

# SHADOWS
lblShadow = tk.Label(window, text='Shadow')
lblShadow.grid(column=0, row=9)

sliderShadow = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL, variable=shadow)
sliderShadow.grid(column=1, row=9)

lblShadowVal = tk.Label(window, text='Shadow = 0')
lblShadowVal.grid(column=2, row=9)

# VIGNETTE
lblVignette = tk.Label(window, text='Vignette')
lblVignette.grid(column=0, row=10)

sliderVignette = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=vignet)
sliderVignette.grid(column=1, row=10)

lblVignetteVal = tk.Label(window, text='Vignette = 0')
lblVignetteVal.grid(column=2, row=10)

# TILT SHIFT
lblTiltShift = tk.Label(window, text='Tilt shift')
lblTiltShift.grid(column=0, row = 11)

sliderTiltShift = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, variable=tilt)
sliderTiltShift.grid(column=1, row=11)

lblTiltShiftVal = tk.Label(window, text='Tilt shift = 0')
lblTiltShiftVal.grid(column=2, row=11)

radioLinear = tk.Radiobutton(window, text='Linear', variable=radio, value='linear')
radioLinear.grid(column=3, row=11)

radioRadial = tk.Radiobutton(window, text='Radial', variable=radio, value='radial')
radioRadial.grid(column=4, row=11)

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
photoLabel.grid(column=5, row=21) # row as placeholder

# Displaying histogram of image
# histogram = ImageTk.PhotoImage(histImg)
# histogramLabel = tk.Label(image=histogram)
# histogramLabel.grid(column=5,row=21) # row as placeholder

window.mainloop()

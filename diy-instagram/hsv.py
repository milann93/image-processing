from PIL import Image
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import math
from rgb import limit


def adjustBrightness(x, coef):

    if coef >= 0:
        coef= 0.5 * coef/100 + 1 # Scales interval [0, 100] to [1, 1.5]
    else:
        coef = 0.5 * (coef + 100)/100 + 0.5 # Scales interval [-100, 0] to [0.5, 1]

    x *= coef
    if x > 255:
        x = 255

    return x


def adjustSaturation(x, coef):

    if coef >= 0:
        coef= 0.5 * coef/100 + 1 # Scales interval [0, 100] to [1, 1.5]
    else:
        coef = 0.5 * (coef + 100)/100 + 0.5 # Scales interval [-100, 0] to [0.5, 1]

    x *= coef
    if x > 255:
        x = 255

    return x


def adjustContrast(v, contrastBoundary):

    contrastBoundary = 1.22*contrastBoundary # range of contrast boundary is [0, 122]

    if v <= contrastBoundary:
        v = 0
    elif v >= 255-contrastBoundary:
        v = 255
    else:
        v = (v - contrastBoundary) / (255 - contrastBoundary) * 255

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


def histogram(hsvimg):
    n = np.arange(0, 256)
    hist = np.zeros(256)
    w, h = hsvimg.size
    npimg = np.array(hsvimg)

    for i in range(w):
        for j in range(h):
            hist[npimg[:, :, 2][j, i]] += 1

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.bar(n, hist)

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pilImg = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return pilImg

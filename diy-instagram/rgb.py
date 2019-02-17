from PIL import Image
import numpy as np


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

    return r, g, b


def adjustFade(r, g, b, fadeCoef):
    # Adds gray image over original
    # fadeCoef is divided by 2 to lower the fading effect

    r += fadeCoef/2
    g += fadeCoef/2
    b += fadeCoef/2

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
    for j in range(w):
        for i in range(h):
            rgbimg[i, j] = np.array([R[i, j], G[i, j], B[i, j]])

    rgbimg = Image.fromarray(rgbimg, 'RGB')
    return rgbimg

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Required, Bug caused by PyQt5
matplotlib.use('Qt5Agg')

def show_cmap(A, ior, image_size, wait=1, title='show_cmap', cmap=cv.COLORMAP_JET):
    if len(A.shape) == 1:
        A = A.reshape((1, -1))

    # Normalize
    A = 128 + (A * 127)
    A = cv.resize(A, image_size)
    A = np.uint8(A)
    # Apply colormap
    img = cv.applyColorMap(A, cv.COLORMAP_JET)

    # Normalize
    ior = 128 + (ior * 127)
    ior = cv.resize(np.expand_dims(ior, axis=0), (image_size[0], 30))
    ior = np.uint8(ior)
    ior = np.resize(ior, (3, *ior.shape)).transpose((1, 2, 0))

    # Stack
    img = np.concatenate((img, ior), axis=0)

    cv.imshow(title, img)
    cv.waitKey(wait)

def show_plot(*args, **kwargs):
    for x in args:
        plt.plot(x)
        plt.draw()

    ylim = kwargs.get('ylim', (-2, 2))
    plt.ylim(*ylim)

    wait = kwargs.get('wait', 1/60)
    plt.pause(wait)

    plt.clf()


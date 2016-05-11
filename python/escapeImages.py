"""
    escapeImages.py

    Contains the escape image generation algortithm as developed by Dan Goodman here:
    https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/

    What I have done is generalize his algorithm to accept any well-behaving lambda and I have added a parameter space
    version of his code.

    This code is specifically being used for a PyMNtos presentation on 5/12/16
"""


import numpy as np
import math
import matplotlib.pyplot as plt

# TODO: Refactor to take a function of two params: first phase, second param (which we will be iterating over).
# Also need to pass the value of the critical point (not always easily infer-able)


def quadMap(phase, param): return phase*phase + param


# TODO: Could use the same function for both phases & param images, would just need to switch parameter orders?

def escImgParam(fName=None, fn=quadMap, critPoint=0, escapeRad=2.0, n=1000, m=1000, itermax=100, xmin=-2, xmax=2,
                ymin=-2, ymax=2, colorMap="nipy_spectral", showAxes=False, interp="None"):
    """
       Creates a 2 dimensional PARAMETER image of whatever function is passed in (as fn)

        :param fName: filename to write to
        :param fn: lambda function representing our current map
            e.g. def fn(phase, param): return phase * phase + param   <--- This would be the quadratic map
        :param critPoint: critical point of fn, sets seed val for repeated iteration
        :param escapeRad: radius of escape for fn (the point beyond which all points escape)
        :param n: resolution of horizontal dimension
        :param m: resolution of vertical dimension
        :param itermax: number of iterations to perform
        :param xmin: horizontal (real) lower bound
        :param xmax: horizontal (real) upper bound
        :param ymin: vertical (imaginary) lower bound
        :param ymax: vertical (imaginary) upper bound
        :param colorMap: string name of Pillow colormap to use
        :return: null, writes image to file
    """

    # makes an n*m grid of integers (for value tracking)
    ix, iy = np.mgrid[0:n, 0:m]

    # n evenly spaced points between xmin and xmax
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]

    c = x + complex(0, 1) * y
    del x, y  # save a bit of memory, we only need z

    img = np.zeros(c.shape, dtype=int)

    ix.shape = n * m
    iy.shape = n * m
    c.shape = n * m

    # Sets all of the z_0 values
    z = fn(critPoint, np.copy(c)) # pycharm warning here, look if something breaks

    for i in range(itermax):

        # If there aren't any values left, stop iterating
        if not len(z):
            break  # all points have escaped

        # Perform an iteration of our map
        z = fn(z, c)

        # A logical vector specifying the points which have escaped
        rem = abs(z) > escapeRad

        # Store the iteration, i, at which the rem points escaped
        img[ix[rem], iy[rem]] = i + 1

        # Now rem represents the points which have remained bounded
        rem = -rem

        # Filter out all escaped values
        z, ix, iy, c = z[rem], ix[rem], iy[rem], c[rem]

    # sets those points which have not yet escaped to itermax + 1 (which is to say that these points took the longest
    #  to escape, or didn't)
    img[img == 0] = itermax + 1

    # Reverses the colormap for aesthetic reasons
    img = abs(itermax - img)

    # create a new figure
    fig = plt.figure()

    image = plt.imshow(img.T, origin='lower left', interpolation=interp)
    ax = plt.gca()
    newXTicks = ["%.2f" % x for x in np.linspace(xmin, xmax, len(ax.get_xticks()) - 1)]
    newYTicks = ["%.2f" % x for x in np.linspace(ymin, ymax, len(ax.get_yticks()) - 1)]
    ax.set_xticklabels([''] + newXTicks) # God knows why I need that empty string at the beginning
    ax.set_yticklabels([''] + newYTicks)

    # Show Axes if needed
    if showAxes:
        plt.axhline(m//2, color='white', linewidth=1)
        plt.axvline(n//2, color='white')

    # Sets the color map if is it is non-empty
    if not colorMap == "":
        image.set_cmap(colorMap)

    # writes to file if a filename is given
    if fName is not None:
        image.write_png(fName + ".png", noscale=True)


def escImgPhase(fName=None, fn=lambda x: quadMap(x, 0), escapeRad=2.0, n=1000, m=1000, itermax=100, xmin=-2, xmax=2,
                ymin=-2, ymax=2, colorMap="nipy_spectral", showAxes=False):
    """
       Creates a 2 dimensional PARAMETER image of whatever function is passed in (as fn)

        :param fName: filename to write to
        :param fn: lambda function representing our current map
            e.g. def fn(phase, param): return phase * phase + param   <--- This would be the quadratic map
        :param critPoint: critical point of fn, sets seed val for repeated iteration
        :param escapeRad: radius of escape for fn (the point beyond which all points escape)
        :param n: resolution of horizontal dimension
        :param m: resolution of vertical dimension
        :param itermax: number of iterations to perform
        :param xmin: horizontal (real) lower bound
        :param xmax: horizontal (real) upper bound
        :param ymin: vertical (imaginary) lower bound
        :param ymax: vertical (imaginary) upper bound
        :param colorMap: string name of Pillow colormap to use
        :return: null, writes image to file
    """
    # makes an n*m grid of integers (for value tracking)
    ix, iy = np.mgrid[0:n, 0:m]

    # n evenly spaced points between xmin and xmax
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]

    seedVals = x + complex(0, 1) * y
    del x, y  # save a bit of memory, we only need z

    img = np.zeros(seedVals.shape, dtype=int)

    ix.shape = n * m
    iy.shape = n * m
    seedVals.shape = n * m

    # Sets all of the z_0 values
    z = fn(seedVals)

    for i in range(itermax):

        # If there aren't any values left, stop iterating
        if not len(z):
            break  # all points have escaped

        # Perform an iteration of our map
        z = fn(z)

        # A logical vector specifying the points which have escaped
        rem = abs(z) > escapeRad

        # Store the iteration, i, at which the rem points escaped
        img[ix[rem], iy[rem]] = i + 1

        # Now rem represents the points which have remained bounded
        rem = -rem

        # Filter out all escaped values
        z, ix, iy = z[rem], ix[rem], iy[rem]

    # sets those points which have not yet escaped to itermax + 1 (which is to say that these points took the longest
    # to escape, or didnt)
    img[img == 0] = itermax + 1

    # Reverses the colormap for aesthetic reasons
    img = abs(itermax - img)

    # create a new figure
    fig = plt.figure()

    image = plt.imshow(img.T, origin='lower left')
    ax = plt.gca()
    newXTicks = ["%.2f" % x for x in np.linspace(xmin, xmax, len(ax.get_xticks()) - 1)]
    newYTicks = ["%.2f" % x for x in np.linspace(ymin, ymax, len(ax.get_yticks()) - 1)]
    ax.set_xticklabels([''] + newXTicks) # God knows why I need that empty string at the beginning
    ax.set_yticklabels([''] + newYTicks)

    # Show Axes if needed
    if showAxes:
        plt.axhline(500, color='white', linewidth=1)
        plt.axvline(500, color='white')

    # Sets the color map if is it is non-empty
    if not colorMap == "":
        image.set_cmap(colorMap)

    # writes to file if a filename is given
    if fName is not None:
        image.write_png(fName + ".png", noscale=True)
import numpy as np
import math
import scipy.ndimage
import matplotlib.image
from PIL import Image
import time
import pylab

def singPertParam(n1, d, beta, angle, n, m, itermax, xmin, xmax, ymin, ymax,filename,colorMap):
    """
    Creates a 2 dimensional PARAMETER image of a singular perturbation of the complex quadratic map
    z = z**n1 + c2 + (beta)/(z.conjugate()**d)
    :param n1: exponent of main phase term
    :param d: exponent of conjugate phase term
    :param beta: perturbation parameter
    :param angle: angle of point from critical circle
    :param n:
    :param m:
    :param itermax:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param filename:
    :param colorMap:
    :return:
    """

    start = time.time()

    #First we need to calculate the critical point that we are going to iterate on:
    r = pow((d/n1)*abs(beta), 1.0/(n1+d))

    #The set of critical points is a circle with the radius given by above, for now we will take the right-most real value on this circle:
    critPoint = r * complex(math.cos(angle), math.sin(angle))


    ix, iy = np.mgrid[0:n, 0:m]

    # n evenly spaced points between xmin and xmax
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]

    c = x+complex(0,1)*y
    del x, y # save a bit of memory, we only need z

    img = np.zeros(c.shape, dtype=int)

    ix.shape = n*m
    iy.shape = n*m
    c.shape = n*m

    #Sets all of the z_0 values
    if critPoint == 0:
        z = critPoint**n1 + np.copy(c)
    else:
        z = critPoint**n1 + np.copy(c) + beta / (critPoint.conjugate()**d)


    for i in range(itermax):
        if not len(z): break # all points have escaped

        #multiply(z, z, z)
        #add(z, c, z)

        z = pow(z, n1) + c + (beta)/(pow(z.conjugate(), d))

        # these are the points that have escaped
        rem = abs(z)>2.0

        img[ix[rem], iy[rem]] = i+1

        rem = -rem

        z = z[rem]
        ix, iy = ix[rem], iy[rem]
        c = c[rem]
    print('Time taken:', time.time() - start)

    img[img==0] = itermax + 1

    image = imshow(img.T, origin='lower left')
    if not colorMap == "":
        image.set_cmap(colorMap)
        image.write_png(filename+'.png', noscale=True)


#Creates a 2 dimensional PHASE image of a singular perturbation of the complex quadratic map
##z = z**n1 + c2 + (beta)/(z.conjugate()**d)
def singPertPhase(n1,d,beta,c2,n, m, itermax, xmin, xmax, ymin, ymax,filename,colorMap):
    start = time.time()

    ix, iy = mgrid[0:n, 0:m]

    x = linspace(xmin, xmax, n)[ix]
    y = linspace(ymin, ymax, m)[iy]

    z = x+complex(0,1)*y
    del x, y # save a bit of memory, we only need z

    img = zeros(z.shape, dtype=int)

    ix.shape = n*m
    iy.shape = n*m
    z.shape = n*m

    #Sets all of the z_0 values
    #z = z**n1 + c2 + (beta)/(z.conjugate()**d)

    for i in range(itermax):
        if not len(z): break # all points have escaped

        #multiply(z, z, z)
        #add(z, c, z)

        z = z**n1 + c2 + (beta)/(z.conjugate()**d)

        # these are the points that have escaped
        rem = abs(z)>2.0

        img[ix[rem], iy[rem]] = i+1

        rem = -rem

        z = z[rem]
        ix, iy = ix[rem], iy[rem]
    print('Time taken:', time.time() - start)

    img[img==0] = itermax + 1

    image = imshow(img.T, origin='lower left')
    if not colorMap == "":
        image.set_cmap(colorMap)
        image.write_png(filename+'.png', noscale=True)


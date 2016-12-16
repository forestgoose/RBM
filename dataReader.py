import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, reshape
from pylab import show, imshow, cm
import math

""" Module made to read the input data of the RBM project """


def loadMinst(labelPath, imagePath, digits = range(9), binary = True, lowDimenson = True):
    """
    Loads MNIST files into 3D numpy arrays. Must give the path of the file
    containing the labels, the images, and the digits you want as a sequence
    of number. The value of digits is set by default to all 10 digits
    if binary is true, the input images are converted into binary images.

    """

    labelsFile = open(labelPath, 'rb') #flbl
    magic_nr, size = struct.unpack(">II", labelsFile.read(8))
    lbl = pyarray("b", labelsFile.read())
    labelsFile.close()

    imageFile = open(imagePath, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", imageFile.read(16))
    img = pyarray("B", imageFile.read())
    imageFile.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    if(lowDimenson == True):
        im = zeros((N, rows//2, cols//2), dtype=uint8)
        for i in range(len(ind)):
            im[i, 0:rows//2, 0:cols //2] = (images[i, 0:rows:2, 0:cols:2]+images[i, 1:rows:2, 0:cols:2]\
            +images[i, 0:rows:2, 1:cols:2]+images[i, 1:rows:2, 1:cols:2])//4
        images = im

    if (binary == True):
        images[images > 1] = 1
        #images[images <= 1] = 0

    return images, labels

def dataMinstToRBM(images, labels):
    ndata, npixelx, npixely = images.shape
    npixel = npixelx * npixely
    data = zeros((ndata, npixel + 10))
    digits = [0,1,2,3,4,5,6,7,8,9]
    z = zeros(10)
    for i in range(ndata):
        temp = zeros(10); temp[labels[i]] = 1
        data[i] = append(images[i].reshape(npixel), temp)
    return data

def lowDimData(images, labels):
    ndata, npixelx, npixely = images.shape
    npixel = npixelx * npixely // 4
    nx = npixelx // 2
    ny = npixely // 2
    print(npixel)
    data = zeros((ndata, npixel + 10))
    digits = [0,1,2,3,4,5,6,7,8,9]
    z = zeros(10)
    for i in range(ndata):
        temp = zeros(10); temp[labels[i]] = 1
        iml = zeros((nx,ny))
        for j in range(nx):
            for k in range(ny):
                iml[j, k] = images[i,2*j, 2*k]
        iml = iml.reshape(nx*ny)
        data[i] = append(iml, temp)
    return data


#test de la lecture si ce fichier est executé en tant que main
if __name__ == "__main__":
    images, labels = loadMinst("Data/train-labels.idx1-ubyte", "Data/train-images.idx3-ubyte")
    imshow(images[0], cmap=cm.gray)
    print(labels[0])
    data = dataMinstToRBM(images, labels)
    print(data[0])
    show()

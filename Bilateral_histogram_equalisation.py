import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def openImageHDR(path):                 #Open HDR open cv Image
    #image/image.tif
    hdr_img = cv.imread(path, -1)
    
    return hdr_img

def showImageHDR(hdr_img):              #Show HDR open cv Image
    cv.imshow('TiffShow',hdr_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def sliceImageHDR(hdr_img):             #Slice Image
    x = np.linspace(0,65535,30)
    slices = []
    for i in range(29):
        new= hdr_img[:,:] < x[i+1]
        new = hdr_img*new
        new1= new[:,:] > x[i]
        slices.append(new1*hdr_img)
    
    return slices

def showPltFig(slices):                 #Show Plt Figures
    columns = 5
    rows = 6
    fig = plt.figure(figsize=(20, 20))
    for i in range(1, columns*rows +1):
        img = slices[i-2]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

def savePltFig(slices,path):            #Save Plt Figure
    columns = 5
    rows = 6
    fig = plt.figure(figsize=(20, 20))
    for i in range(1, columns*rows +1):
        img = slices[i-2]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.savefig(path)

def stretchHist(slc):                   
    max_bit = 2**16

    hist,bins = np.histogram(slc.flatten(),max_bit,[0,max_bit])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*max_bit/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint16')

    return cdf[slc]

def stretchSlicesHist(slices):
    slices_stretch = []
    for slc in slices :
        slc_stretch = stretchHist(slc)
        slices_stretch.append(slc_stretch)
    
    print(np.array(slices_stretch).shape) # Show shape
    return slices_stretch

    

    
if __name__ == "__main__":

    hdr_img = openImageHDR("image/image.tif")

    #showImageHDR(hdr_img)

    slicedImage = sliceImageHDR(hdr_img)

    #showPltFig(slicedImage)
    savePltFig(slicedImage,'image/SlicedStretchImages')

    slicedStretchImage = stretchSlicesHist(slicedImage)

    #showPltFig(slicedStretchImage)
    savePltFig(slicedStretchImage,'image/SlicedStretchImages')

    showImageHDR(slicedImage[5])
    showImageHDR(slicedStretchImage[5])
    





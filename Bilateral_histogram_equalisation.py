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
    sk = []
    for i in range(29):
        null_compare = hdr_img[:,:] >= x[i+1]
        if i > 0: 
            slice = (hdr_img - x[i]) / (x[i+1] - x[i])
            slice = slice*null_compare
            m, n = np.where(hdr_img >= x[i])
            slice[m,n] = hdr_img[m, n]
        else:
            slice = hdr_img
        sk.append(slice)
    return sk

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

    return cdf[slc.astype('uint16')]

def stretchSlicesHist(slices):
    slices_stretch = []
    for slc in slices :
        slc_stretch = stretchHist(slc)
        slices_stretch.append(slc_stretch)
    
    print(np.array(slices_stretch).shape) # Show shape
    return slices_stretch


def fusionSlices(stretched_slices,wi):
    i= 0
    fusion = wi[i]*stretched_slices[i]
    for slc in stretched_slices :
        if i == 0 :
            fusion = fusion
        else :
            fusion += (np.matrix.dot(slc,wi[i])).astype('uint16')
        i+=1
    return fusion

def getWeightMap():
    x = np.linspace(0,65535,30)
    wk = []
    for i in range(29):
        one_compare = hdr_img[:,:] <= x[i]
        null_compare = hdr_img[:,:] >= x[i+1]
        other_compare = (~ one_compare)*(~ null_compare)

        wk.append(other_compare*hdr_img)
    return wk
    
def fusionNiko(wk, sk):
    output = []
    for i in range(29):
        print(f"Processing slice {i}/29, progressing: {100*i/29}%", end="\r")
        output.append(wk[i] @ sk[i])
    print("Done                                                  ")
    result = output[0]
    for i in range(1, 29):
        result += output[i].astype('uint16')
    
    return result


if __name__ == "__main__":

    hdr_img = openImageHDR("image/image.tif")

    #showImageHDR(hdr_img)

    slicedImage = sliceImageHDR(hdr_img)

    #showPltFig(slicedImage)
    # savePltFig(slicedImage,'image/SlicedStretchImages')

    #slicedStretchImage = stretchSlicesHist(slicedImage)

    #showPltFig(slicedStretchImage)
    # savePltFig(slicedStretchImage,'image/SlicedStretchImages')

    strechHist = stretchSlicesHist(slicedImage)
    # print(strechHist[5].shape)
    weights = getWeightMap()

    # fusion = fusionSlices(strechHist,weights)
    fusion = fusionNiko(weights, strechHist)
    # print(fusion)
    showImageHDR(fusion)

    





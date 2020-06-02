# import library
import  cv2
import numpy as np

# global function
#---------------RESIZE IMAGE FUNCTION
def resizeImage(img,percent):
    width_resize =int (img.shape[1]*percent/100)
    height_resize =int(img.shape[0]*percent/100)
    dim= (height_resize,width_resize)
    imgResize =cv2.resize(img,dim ,cv2.INTER_AREA)
    return imgResize

#---------------SELECT BIGEST REGION
def selectBigestRegion(mask):
    mask = mask.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    return img2 
#---------------SELECT REGION BASE ON AREA
def selectRegion(mask,minValue,maxValue) :
    mask = mask.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]
    thisList=[]
    for i in range(2, nb_components):
            if sizes[i] > minValue and sizes[i] < maxValue:
                thisList.append(i)
    
    img2 = np.zeros(output.shape)
    img =np.zeros(output.shape)
    for i in range(0,len(thisList)-1):       
        img2[output == thisList[i]] = 255
       
    cv2.imshow("Select Area Region :"+  str(minValue)+"=>"+str(maxValue), img2)
    return img2 


# --------------------------------------MAIN--------------------------------------

img =cv2.imread('code.jpeg')
imgResize =resizeImage(img,20)
# Gray
imgGray =cv2.cvtColor(imgResize,cv2.COLOR_RGB2GRAY)
ret,imgThreshold= cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY)
kernel=np.ones((10,10),np.uint8)
imgThreshold =cv2.morphologyEx(imgThreshold,cv2.MORPH_CLOSE ,kernel)
kernel=np.ones((5,5),np.uint8)
imgThreshold =cv2.morphologyEx(imgThreshold,cv2.MORPH_OPEN ,kernel)
# 
minArea =10
maxArea=10000
connectivity=4
output = cv2.connectedComponentsWithStats(imgThreshold, connectivity, cv2.CV_32S)
            
for i in range(output[0]):
            if output[2][i][4] >= minArea and output[2][i][4] <= maxArea:
                cv2.rectangle(imgResize, (output[2][i][0], output[2][i][1]), (
                    output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)


#selectBigestRegion(imgThreshold)
selectRegion(imgThreshold,2,50)

#--------------- IMSHOW RESULT
# cv2.imshow('detection', imgResize)
cv2.imshow('Threshold',imgThreshold)
cv2.waitKey(0)

# import library
import  cv2
import numpy as np
import matplotlib.pyplot as plt

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
    return img2 
#---------------SELECT REGION BASE ON AREA
def selectRegion(mask,option,minValue,maxValue) :
    if  option== "area":       
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
        
        #cv2.imshow("Select Area Region :"+  str(minValue)+"=>"+str(maxValue), img2)
    elif option=="width":
        mask = mask.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        sizes = stats[:, 2]
        thisList=[]
        for i in range(2, nb_components):
                if sizes[i] > minValue and sizes[i] < maxValue:
                    thisList.append(i)
        
        img2 = np.zeros(output.shape)
        img =np.zeros(output.shape)
        for i in range(0,len(thisList)-1):       
            img2[output == thisList[i]] = 255
        
        #cv2.imshow("Select Area Region :"+  str(minValue)+"=>"+str(maxValue), img2)
    elif option=="height":
        mask = mask.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        sizes = stats[:, 3]
        thisList=[]
        for i in range(2, nb_components):
                if sizes[i] > minValue and sizes[i] < maxValue:
                    thisList.append(i)
        
        img2 = np.zeros(output.shape)
        img =np.zeros(output.shape)
        for i in range(0,len(thisList)-1):       
            img2[output == thisList[i]] = 255
        
        #cv2.imshow("Select Area Region :"+  str(minValue)+"=>"+str(maxValue), img2)
       
    return img2 


  
# --------------------------------------MAIN--------------------------------------
img =cv2.imread('C:/Users/manmu/Downloads/git_train/Python_OpenCV_ImageProcessing/code.jpeg')
imgResize =resizeImage(img,20)
# Gray
imgGray =cv2.cvtColor(imgResize,cv2.COLOR_RGB2GRAY)
ret,imgThreshold= cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY)
kernel=np.ones((10,10),np.uint8)
imgThreshold =cv2.morphologyEx(imgThreshold,cv2.MORPH_CLOSE ,kernel)
kernel=np.ones((5,5),np.uint8)
imgThreshold =cv2.morphologyEx(imgThreshold,cv2.MORPH_OPEN ,kernel)
# 
connectivity=4
output = cv2.connectedComponentsWithStats(imgThreshold, connectivity, cv2.CV_32S)
minArea =200
maxArea=10000        
for i in range(output[0]):
            if output[2][i][4] >= minArea and output[2][i][4] <= maxArea:
                cv2.rectangle(imgResize, (output[2][i][0], output[2][i][1]), (
                    output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)


#selectBigestRegion(imgThreshold)
option_shape=["area","width","height"]
method=option_shape[0]
regionSelected=selectRegion(imgThreshold,method,300,400)
regionBigest =selectBigestRegion(imgThreshold)

#--------------- IMSHOW RESULT

#cv2.imshow('Threshold',imgThreshold)
#cv2.imshow('Region selected '+str( option_shape[2]),regionSelected)
cv2.waitKey(0)
fig, axit =plt.subplots(1,3)
axit[0].imshow(imgThreshold,cmap='gray')
axit[0].set_title('image Threshold')

axit[1].imshow(regionSelected,cmap='gray')
axit[1].set_title("region Selected "+str(method))

axit[2].imshow(regionBigest,cmap='gray')
axit[2].set_title("region Biggest")


fig.tight_layout()
plt.show()
import cv2
import numpy as np

# The following code states the path of he image to be processed
path=str('/home/jithin/Desktop/Perception /HW1/for_watson.png')

# Now we have to convert the image to a NumPy array, So we use the Open Cv function imread()
matrixedimg=cv2.imread(path)
# print(matrixedimg)
print("Dimensions of the image:", matrixedimg.shape)  #Gives: Dimensions of the image: (480, 640, 3)

# Now to apply thresholding lets extract unique grayscale intensities from the generated array
grayscaled=cv2.cvtColor(matrixedimg,cv2.COLOR_BGR2GRAY) 
print(grayscaled)
print("Dimensions of the image:", grayscaled.shape)   #Gives: Dimensions of the image: (480, 640)


#   |cv2.imshow('Grayscale Image', grayscaled)
#   |cv2.waitKey(0)  # Waits for a key press to close the window
#   |cv2.destroyAllWindows()  # Closes the windows
#Viewing the grayscaled image, a part of sherlocks code is visualisable but that is not enough so we 
#will interchange contrasting values to differentiate the remainder of the code concealed (0W-255B)

grayscaled[grayscaled==29]=255
grayscaled[grayscaled==150]=0
grayscaled[grayscaled==76]=0

cv2.imshow('Grayscale Image', grayscaled)
cv2.waitKey(0)  # Waits for a key press to close the window
cv2.destroyAllWindows()  # Closes the windows
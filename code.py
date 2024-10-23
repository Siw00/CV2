#1. Edge detection 
#!pip install google-colab

from google.colab.patches import cv2_imshow
import cv2

image= cv2.imread('/content/tiger.jpg')

print("orgignal image \n")
cv2_imshow(image)

cv2.imwrite('/content/tiger.png',image)

image= cv2.imread('/content/tiger.jpg')

#edgedetection
cv2.imwrite('/content/edge_tiger.jpg',cv2.Canny(image,200,300))

image= cv2.imread('/content/edge_tiger.jpg')
print("Edge image \n")
cv2_imshow(image)




#2 Face detection 
from google.colab.patches import cv2_imshow
import cv2

img = cv2.imread('/content/face.jpg')
classifier = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face = classifier.detectMultiScale(gray_img)

for x,y,w,h in face:
  cv2.rectangle(img,(x,y),(x+w,y+h),(234,45,66),2)

cv2_imshow(img)






#3. OCR
import pytesseract
from PIL import Image

image_path = "/content/handwriting.png"

image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print(text)







#4. Dilation opening closing Erosion
import cv2
import numpy as np
from google.colab.patches import cv2_imshow 
cv2.waitKey(0)
img = cv2.imread('/ content/tiger.jpg', 0)
kernel = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2_imshow(img)
cv2_imshow(img_erosion)
cv2_imshow(img_dilation)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2_imshow(opening)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2_imshow(closing)








#5. Image cropping
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/tiger.jpg")
# Check the type of read image

print(type(img))

# Check the shape of the input image
print("Shape of the image:", img.shape)

# [rows, columns]
crop = img[80:280, 150:330]

cv2_imshow(img)
cv2_imshow(crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Import the library
import cv2

# Using Haal Cascade classifier to train the model
face_cascade=cv2.CascadeClassifier("datasets/haarcascade_frontalcatface.xml")

# Input Image as a test
image=cv2.imread("Image/cat_dog_image2.jpg")

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cat_face=face_cascade.detectMultiScale(gray,1.1,1)

for (x,y,w,h) in cat_face:
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    
image=cv2.resize(image,(800,700))
cv2.imshow("Face Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()    
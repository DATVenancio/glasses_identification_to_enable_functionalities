from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detectGlasses(frame,mask_model):
    faces =[]
    preds =[]
    (frame_height,frame_width)=frame.shape[:2]
    face=frame[0:]
    face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    face = cv2.resize(face,(224,224))
    face=img_to_array(face)
    face=preprocess_input(face)

    faces.append(face)
    faces=np.array(faces,dtype="float32")

    preds = mask_model.predict(faces,batch_size=32)
    return preds

def writeResults(frame,glasses_prediction):
    text_background_color = (0,0,0)
    cv2.rectangle(frame, (400, 10), (800, 60),text_background_color, thickness=-1)

    
    (mask,without_mask) = glasses_prediction
    text_message = "GLASSES" if mask>without_mask else "NO GLASSES"
    text_color = (0, 255, 0) if mask>without_mask else (0,0,255)
    
    cv2.putText(frame, text_message, (450,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,text_color,4)

    
def displayCamera():
    camera_frame = video_controller.read()
    camera_frame = imutils.resize(camera_frame,width=1200)
    

    return camera_frame

def displaySecretImage(glasses_prediction):
    (glasses,no_glasses) = glasses_prediction
    image_text = "message2.png" if glasses>no_glasses else "message1.png"
    secret_image = cv2.imread(image_text,cv2.IMREAD_ANYCOLOR)
    secret_image = imutils.resize(secret_image,width=700)
    cv2.imshow("SecretImage",secret_image)
    cv2.moveWindow("SecretImage",1400,300)

glasses_detector_model = load_model("glasses_detector.model")
video_controller = VideoStream(src=0).start()
teste=0
while True:
    camera_frame=displayCamera()
    glasses_prediction = detectGlasses(camera_frame,glasses_detector_model)[0]
    writeResults(camera_frame,glasses_prediction)
    displaySecretImage(glasses_prediction)
    
    cv2.imshow("Frame",camera_frame)
    cv2.moveWindow("Frame",0,0)
    key = cv2.waitKey(1) & 0xFF
    camera_frame = cv2.add
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video_controller.stop()
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
import time

label = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4,'trash': 5}

MODEL_PATH = 'vgg16-0014.hdf5'
model = load_model(MODEL_PATH)

def model_predict(img_path):
    
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    y_classes = preds.argmax()
    result = str(list(label.keys())[list(label.values()).index(y_classes)])
    return result

capture_duration = 120
cam = cv2.VideoCapture(0)
start_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (225, 0, 0)
thickness = 2
while(int(time.time() - start_time) < capture_duration):

    success,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if success:
        cv2.imwrite('image.jpg',img)
        result = model_predict('image.jpg')
        # print(result)
    image1 = cv2.putText(img, result, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Garbage",image1)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
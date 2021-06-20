import cv2
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
import tensorflow.keras
import math


np.set_printoptions(suppress=True)
# initialize face and pose detection
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# initalize draw
mp_drawing = mp.solutions.drawing_utils
# import model
model = tensorflow.keras.models.load_model('/Users/Annie/Desktop/keras_model.h5')
# activate video capture using webcam
cap = cv2.VideoCapture(0)
# sets the number of images the model can process at a time, first number in shape() is the # of images
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# initialize variables for later usage
nose = [0, 0] # nose coordinates for pose detection
mouth = [0, 0] # mouth coordinates for pose detection
countFace = [] # buffer count
countPose = 0
mask = '' # displas mask or no mask

with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection: # confidence level of face detection, default is 0.5
    while True:
        success, image = cap.read() # captures image oon webcam
        image.flags.writeable = False #disables draw, for smoother face detection
        results = face_detection.process(image) # use face detection on image
        result2 = pose.process(image)   #use pose detection on image
        image.flags.writeable = True # enables draw

        # face detection chunk
        if results.detections: # if detect face
            for id, detection in enumerate(results.detections):
                countFace.append(0)
                # bounding box
                bbox1 = detection.location_data.relative_bounding_box
                height, width, channel = image.shape
                # calculations and setting coordinates
                bbox = int((bbox1.xmin * width)), int((bbox1.ymin * height)), \
                       int(bbox1.width * width), int(bbox1.height * height)
                center = (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[3]/2)))
                r = (center[0] - bbox[2])
                t = (center[1] - bbox[3])
                l = (center[0] + bbox[2])
                b = (center[1] + bbox[3])
                video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # creates new "image" from image that is RBG for processing
                im = Image.fromarray(video, 'RGB') # further processes the image
                im = im.crop((r, t, l, b)) # crops about the area around the head
                im = im.resize((224, 224)) # resizes for model for better dectection
                img_array = np.asarray(im) # turns image into array for the model
                img_array = np.expand_dims(img_array, axis=0) # further processes the array for the model
                final = (img_array.astype(np.float32) / 127.0) - 1 # more processing and smoothening
                prediction = model.predict(final) # inputs the processed array into the model and get the result
                if prediction[0][0] >= prediction[0][1]: # if more likely for to be mask than no mask
                    if countFace[id] < 5:
                        countFace[id] += 1
                    if countFace[id] > 3:
                        mask = 'Mask F'
                else: # if more likely for to be no mask than mask
                    if countFace[id] > -5:
                        countFace[id] -=1
                    if countFace[id] < -3:
                        mask = 'No Mask F'
                cv2.putText(image, mask, (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, # display mask or no mask
                            3, (0, 0, 255), 3)

        # pose detection chunk
        elif result2.pose_landmarks: # if pose detection detects
            for id, lm in enumerate(result2.pose_landmarks.landmark):
                h, w, c = image.shape #gets height and width
                cx, cy = int(lm.x * w), int(lm.y * h) # get the x and y coords of the landmarks
                if id == 0: # if id is a nose
                    nose = [cx, cy] # gets the x and y coord of nose
                if id == 10: # if id is a mouth
                    mouth = [cx, cy] # get the x and y coord of the mouth (right side)
            leng = math.hypot(nose[0] - mouth[0], nose[1] - mouth[1]) # gets distance between mouth and nose
            r = nose[0] - (5 * leng)
            t = nose[1] - (5 * leng)
            l = nose[0] + (5 * leng)
            b = nose[1] + (4 * leng)
            # following code is identical to face detection
            video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(video, 'RGB')
            im = im.crop((r, t, l, b))
            im = im.resize((224, 224))
            img_array = np.asarray(im)
            img_array = np.expand_dims(img_array, axis=0)
            final = (img_array.astype(np.float32) / 127.0) - 1
            prediction = model.predict(final)
            if prediction[0][0] >= prediction[0][1]:
                if countPose < 5:
                    countPose += 1
                if countPose > 3:
                    mask = 'Mask P'
            else:
                if countPose > -5:
                    countPose -= 1
                if countPose < -3:
                    mask = 'No Mask P'
            cv2.putText(image, mask, (int(nose[0]), int(nose[1])), cv2.FONT_HERSHEY_PLAIN,
                                     3, (0, 0, 255), 3)

        # when no face or pose is detected
        else:
            #resets counter when face is not detected
            if countPose > 0:
                countPose -= 1
            if countPose < 0:
                countPose += 1
            for i in range(0, len(countFace)):
                if countFace[i] > 0:
                    countFace[i] -= 1
                if countFace[i] < 0:
                    countFace[i] += 1
            print("No face")

        cv2.imshow("Image", image) #display
        key = cv2.waitKey(1)
        if key == 27: #if key 'esc' is pressed, close the program
            cap.release()
            cv2.destroyAllWindows()
            exit()
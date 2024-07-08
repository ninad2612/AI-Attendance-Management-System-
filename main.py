import cv2
import numpy as np
import face_recognition
import os

# Load face recognition model and known faces
student_dict = {
    "Vedant": 1,
    "Prathmesh": 2,
    "Dhrub": 3,
    "Siddhesh": 4
}

path = 'F:/Attendence using Face Recognition/images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    if len(facesCurFrame) > 1:
        message = "Two persons detected. Only one can be detected."
        
        textSize = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        textX = (img.shape[1] - textSize[0]) // 2
        textY = 30
        cv2.putText(img, message, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif len(facesCurFrame) == 0:
        cv2.putText(img, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    else:
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                detected_name = classNames[matchIndex]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, detected_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        
    cv2.imshow('Webcam', img)
    
    # Check for the escape key (ASCII value 27)
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

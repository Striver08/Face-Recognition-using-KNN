import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

capture = cv.VideoCapture(0)

skip = 0
face_data  = []
dataset_path = './data/'
file_name = input("Please enter your name: ")


while True:
    isTrue , frame = capture.read()

    faces = face_cascade.detectMultiScale(frame,1.1,1)
    faces = sorted(faces, key= lambda f:f[2]*f[3])
    #print(faces)
    for face in faces[-1:]: 
        x,y,w,h = face
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
    
        #Extracting the Region of Interest i.e. Cropping out the reqd. frame
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv.resize(face_section,(100,100))

        skip +=1 
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

        cv.imshow("Face Recognition", frame)

        cv.imshow("Frame",face_section)

    if cv.waitKey(1) & 0xFF == ord('w'):
        break

#Converting faceList(face_data) into numpy array.

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(dataset_path+file_name)
np.save(dataset_path+file_name,face_data)
print(file_name+" has been succesfully saved. in "+ dataset_path+file_name+'.npy')

capture.release()
cv.destroyAllWindows()
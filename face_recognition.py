
import cv2 as cv
import numpy as np
import os

def distance():
    #Euclidian 
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k =5):
    dist = []

    for i in range(train.shape[0]):
        #Get the vector and label 
        ix = train[i, :-1]
        iy = train[i, :-1]
        # Compute the distance from the test point
        d = distance(test,ix)
        dist.append([d,iy])
    #Sort based on the distance and get top k
    dk = sorted(dist,key=lambda x:x[0])[:k]
    #Retrieve only the labels
    labels = np.array(dk)[:,-1]

    #Get frequencies of each label
    output = np.unique(labels, return_counts=True)

    #Find max frequency and correspondnig label
    index = np.argmax(output[1])
    
    return output[0][index]
################################


face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

capture = cv.VideoCapture(0)

skip = 0
dataset_path = './data/'

face_data = []
label = []
class_id = 0 #Label for the given file
names = {}   #Mapping between id-name

################################

#Data Preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print("Loaded"+ fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id +=1
        label.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels), axis=1)
print(trainset.shape)

#Testing

    while True:
        isTrue, frame = capture.read()







from imutils import paths
import face_recognition
import imutils
import pickle
import cv2
import os
print("quantifying faces...")
imagePaths = list(paths.list_images('dataset'))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
 print("processing image {}/{}".format(i + 1,len(imagePaths)))
 name = imagePath.split(os.path.sep)[-2]
 image = cv2.imread(imagePath)
 rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 boxes = face_recognition.face_locations(rgb,)
 encodings = face_recognition.face_encodings(rgb, boxes)
 for encoding in encodings:
  knownEncodings.append(encoding)
  knownNames.append(name)
 print(name) 
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
print(data);
pickle_out = open("model.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()
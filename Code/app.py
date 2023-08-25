import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\Admin\Documents\GitHub\Face-Recognition-with-CNN\Model\Face Recognizer.h5')
class_names = {
    0: 'Bill Gates',
    1: 'Elon Musk',
    2: 'Jeff Bezos',
    3: 'Mark Zuckerberg',
    4: 'Steve Jobs',
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        prediction = model.predict(np.expand_dims(face, axis=0))
        recognized_person_index = np.argmax(prediction)
        
        highest_probability = prediction[0][recognized_person_index]
        
        recognition_threshold = 0.85
        
        if highest_probability >= recognition_threshold:
            
            recognized_person_name = class_names.get(recognized_person_index)
            text = f"Recognized Person: {recognized_person_name}"
        else:
            text = "Person Not Registered"

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

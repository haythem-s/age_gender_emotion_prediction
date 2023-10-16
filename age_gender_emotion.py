# Import Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st

# Paths for face detection and gender/age/emotion prediction models
GENDER_MODEL = 'deploy_gender.prototxt'
GENDER_PROTO = 'gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

FACE_PROTO = "deploy.prototxt.txt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

AGE_MODEL = 'deploy_age.prototxt'
AGE_PROTO = 'age_net.caffemodel'
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Load the emotion model
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)

# Load the model weights
emotion_model.load_weights('emotion_model_weights.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

frame_width = 1280
frame_height = 720

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []

    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()

def get_emotion_predictions(face_img):
    # Preprocess the image for the emotion model
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0  # Normalize the image

    # Make predictions using the Keras emotion model
    emotion_preds = emotion_model.predict(face_img)
    return emotion_preds

def predict_age_gender_and_emotion():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    quit_button = st.button("Quit")
    unique_key = 0

    while not quit_button:
        _, img = cap.read()
        frame = img.copy()
        
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        
        faces = get_faces(frame)
        
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            
            age_preds = get_age_predictions(face_img)
            gender_preds = get_gender_predictions(face_img)
            emotion_preds = get_emotion_predictions(face_img)

            i = emotion_preds[0].argmax()
            emotion = emotion_labels[i]
            emotion_confidence_score = emotion_preds[0][i]

            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]

            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]
            
            label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%, Emotion: {emotion}-{emotion_confidence_score*100:.1f}%"
            yPos = start_y - 15
            
            while yPos < 15:
                yPos += 15
            
            box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            # Define the font and font scale
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.54

            # Calculate the position for the gender/age information
            text_x = start_x
            text_y = yPos -5

            # Draw the gender/age information
            cv2.putText(frame, f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%",
                        (text_x, text_y), font, font_scale, box_color, 2)

            # Calculate the position for the emotion information below the gender/age information
            text_y_emotion = text_y -20  # Adjust the vertical position here

            # Draw the emotion information
            cv2.putText(frame, f"Emotion: {emotion}-{emotion_confidence_score*100:.1f}%",
            (text_x, text_y_emotion), font, font_scale, box_color, 2)

        
        stframe.image(frame, channels="BGR")
    
    cap.release()

def main():
    predict_age_gender_and_emotion()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
import tempfile

# Paths for face detection and gender/age/emotion prediction models
FACE_PROTO = "deploy.prototxt.txt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_MODEL = 'deploy_gender.prototxt'
GENDER_PROTO = 'gender_net.caffemodel'
AGE_MODEL = 'deploy_age.prototxt'
AGE_PROTO = 'age_net.caffemodel'
EMOTION_MODEL_JSON = 'emotion_model.json'
EMOTION_MODEL_WEIGHTS = 'emotion_model_weights.h5'
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


frame_width = 1280
frame_height = 720

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Load emotion prediction model
with open(EMOTION_MODEL_JSON, 'r') as json_file:
    emotion_model_json = json_file.read()
emotion_model = model_from_json(emotion_model_json)
emotion_model.load_weights(EMOTION_MODEL_WEIGHTS)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)

def get_emotion_predictions(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0  # Normalize the image

    emotion_preds = emotion_model.predict(face_img)
    return emotion_preds

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

def predict_age_gender_and_emotion(image_file):
    # Read Input Image
    img = cv2.imread(image_file)

    # Resize the image if needed
    if img.shape[1] > frame_width:
        img = image_resize(img, width=frame_width)

    # Predict faces
    faces = get_faces(img)
    
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        # Extract face image
        face_img = img[start_y: end_y, start_x: end_x]

        # Predict age and gender
        age_preds = get_age_predictions(face_img)
        gender_preds = get_gender_predictions(face_img)

        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]

        # Predict emotion
        emotion_preds = get_emotion_predictions(face_img)
        i = emotion_preds[0].argmax()
        emotion = emotion_labels[i]
        emotion_confidence_score = emotion_preds[0][i]

        # Display gender, age, and emotion information as text
        st.text(f"Gender: {gender}")
        st.text(f"Age: {age}")
        st.text(f"Emotion: {emotion}")
        st.text("")

        # Draw the box and labels
        yPos = start_y - 15

        while yPos < 15:
            yPos += 15

        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 2)

        font_scale = 0.5
        # Define the font and font scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.54

# Calculate the position for the gender/age information
        text_x = start_x
        text_y = yPos -5

# Draw the gender/age information
        cv2.putText(img, f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%",
                    (text_x, text_y), font, font_scale, box_color, 2)

# Calculate the position for the emotion information below the gender/age information
        text_y_emotion = text_y -20  # Adjust the vertical position here

# Draw the emotion information
        cv2.putText(img, f"Emotion: {emotion}-{emotion_confidence_score*100:.1f}%",
                     (text_x, text_y_emotion), font, font_scale, box_color, 2)


    # Display and save the image
    st.image(img, caption="Output Image with Age, Gender, and Emotion Detection", use_column_width=True, channels="BGR")

def load_image_and_predict(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(uploaded_image.read())

    predict_age_gender_and_emotion(temp_image.name)

def main():
    load_image_and_predict()

if __name__ == "__main__":
    main()

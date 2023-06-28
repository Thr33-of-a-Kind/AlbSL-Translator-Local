import os.path
import pickle
import warnings
import xgboost
import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# this script is to run Streamlit locally
warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("AlbSL Translator")


@st.cache_data
def format_labels(option):
    option = str(option)
    option = option.replace('_', ' ')
    option = option.title()
    return option


model_names = ["random_forest", "svm", "knn"]
model_option = st.selectbox(
    'What model do you want to use ?',
    model_names,
    format_func=format_labels
)

absolute_path = os.path.dirname(__file__)


@st.cache_resource
def load_labels():
    labels_path = absolute_path + "/labels.txt"
    labels = {}
    with open(labels_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                index, predicted_character = line.split(' ')
                labels[int(index)] = predicted_character

    return labels


@st.cache_resource
def load_models():
    models = {}

    for model_name in model_names:
        model_path = absolute_path + '/' + model_name + '.pkl'

        with open(model_path, 'rb') as file:
            pickles = pickle.load(file)

        models[model_name] = pickles['model']

    return models


labels = load_labels()
models = load_models()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def callback(frame):
    image = frame.to_ndarray(format="bgr24")

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dataAux = []
    xList = []
    yList = []

    height, width, _ = image.shape

    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for _, handLandmark in enumerate(handLandmarks.landmark):
                x = handLandmark.x
                y = handLandmark.y
                xList.append(x)
                yList.append(y)

            for _, handLandmark in enumerate(handLandmarks.landmark):
                x = handLandmark.x
                y = handLandmark.y
                dataAux.append(x - min(xList))
                dataAux.append(y - min(yList))

        x1 = int(min(xList) * width) - 10
        y1 = int(min(yList) * height) - 10

        x2 = int(max(xList) * width) - 10
        y2 = int(max(yList) * height) - 10

        dataAux = np.pad(dataAux, (0, 84 - len(dataAux)))

        model = models[model_option]

        prediction = model.predict([np.asarray(dataAux)])
        predicted_character = labels[int(prediction[0])]

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="AlbSL Translator",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True},
    video_frame_callback=callback,
    async_processing=True
)

import streamlit_2 as st

import tensorflow as tf
from load_model_new import load_model_raton
import cv2
# Load the model
st.title("App Medical prediction for cardiacal sicks!")
model = tf.keras.models.load_model_raton()

model.summary

# Define the input and output of the model
inputs = tf.keras.Input(shape=(224, 224, 3))
outputs = model(inputs)

# Write a function to run the model
def predict(input_data):
  return model.predict(input_data)

# Read the video
video = cv2.VideoCapture("video.mp4")

# Iterate through the frames
while video.isOpened():
  # Extract the frame
  success, frame = video.read()

  # Preprocess the frame
  frame = cv2.resize(frame, (224, 224))
  frame = frame / 255.0
  frame = tf.expand_dims(frame, 0)

  # Run the model on the frame
  prediction = predict(frame)

  # Display the output
  st.image(frame)
  st.write("Prediction: ", prediction)

video.release()
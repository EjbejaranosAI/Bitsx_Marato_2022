import cv2
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import os



def load_model_raton(): # Canviar paths!!
  #Loading model and weights
  json_file = open('/content/drive/MyDrive/bitsxlaMarato/95-1003-995_W2_lr0.05model.json','r')
  model_json = json_file.read()
  json_file.close()
  model = model_from_json(model_json)
  model.load_weights('/content/drive/MyDrive/bitsxlaMarato/95-1003-995_W2_lr0.05weights.hdf5')
  return model


from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from config import *

def carregar_modelo():
    model = load_model(modelo_path)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

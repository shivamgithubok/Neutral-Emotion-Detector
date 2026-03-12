import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

from tensorflow.keras.models import load_model
model = load_model("best_model.h5", compile=False)
print("✓ Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
pred = model.predict(tf.zeros((1,224,224,3)), verbose=0)
print("✓ Model prediction successful!")
print("Prediction shape:", pred.shape)
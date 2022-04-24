from pyexpat import model
import tensorflow as tf
from layers import BilinearUpSampling2D


custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D,
                  'depth_loss_function': None}

h5model = tf.keras.models.load_model('nyu.h5', custom_objects=custom_objects)

tf.saved_model.save(h5model, "depthmodel")

model = tf.saved_model.load("depthmodel")

concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 480, 640, 3])

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations  = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('depthmodel.tflite', 'wb') as f:
  f.write(tflite_model)
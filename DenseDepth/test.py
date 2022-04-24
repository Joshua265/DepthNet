import tensorflow as tf

imported_with_signatures = tf.saved_model.load("depthmodel")
print(imported_with_signatures.signatures["serving_default"])
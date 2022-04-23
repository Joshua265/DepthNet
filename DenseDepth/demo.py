import time
import numpy as np
import cv2
import tensorflow as tf
from layers import BilinearUpSampling2D


def DepthNorm(x, maxDepth):
    return maxDepth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=1):
   
    images = np.resize(images, (1, 480, 640, 3))/255
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D,
                  'depth_loss_function': None}

model = tf.keras.models.load_model('nyu.h5', custom_objects=custom_objects)

cap = cv2.VideoCapture(1)


counter = 0
start_time = time.time()
while True:
    print("predicht new image")
    ret, raw_image = cap.read()  # read one frame
    input_image = raw_image.astype(np.float32)

    pred = predict(model, np.array(input_image))
    # print(pred[0])

    rgb_frame = cv2.applyColorMap(
        (pred[0]*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow("image", raw_image)
    cv2.imshow("depth", rgb_frame)
    # cv2.waitKey(0)
    

    counter = counter + 1
    if counter == 30:
        print(f"FPS: {1/((time.time() - start_time)/30)}")
        start_time = time.time()

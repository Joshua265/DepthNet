import cv2
import tensorflow as tf
import time
import numpy as np

def DepthNorm(x, maxDepth):
    return maxDepth / x

minDepth=10
maxDepth=1000

interpreter = tf.lite.Interpreter("depthmodel.tflite")
interpreter.allocate_tensors()

cap = cv2.VideoCapture(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

counter = 0

while True:
    print("predict new image")
    ret, raw_image = cap.read()  # read one frame
    input_image = np.array(raw_image).astype(np.float32)

    images = np.resize(np.stack(input_image), (1, 480, 640, 3))/255

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], images)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred = np.clip(DepthNorm(pred, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
    
    

    # rgb_frame = cv2.applyColorMap(
    #     (pred[0]*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    # cv2.imshow("image", raw_image)
    # cv2.imshow("depth", rgb_frame)
    # cv2.waitKey(0)
    
    print(f"Time: {(time.time() - start_time)}s")
    break
    # counter = counter + 1
    # if counter == 30:
    #     print(f"FPS: {1/((time.time() - start_time)/30)}")
    #     start_time = time.time()

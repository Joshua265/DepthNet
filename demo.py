import cv2
import torch
from NewCRFDepth import NewCRFDepth
import numpy as np
import time

max_depth = 10

model = NewCRFDepth(version='large07', inv_depth=False, max_depth=max_depth)
model = torch.nn.DataParallel(model)

checkpoint = torch.load("model_nyu.ckpt")
model.load_state_dict(checkpoint['model'])
model.eval()
cap = cv2.VideoCapture(1)

while True:

    ret, raw_image = cap.read() #read one frame
    input_image = raw_image.astype(np.float32)

    print("calculating depth information")
    start_time = time.time()

    # Normalize image
    input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
    input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
    input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017


    input_images = np.expand_dims(input_image, axis=0)
    input_images = np.transpose(input_images, (0, 3, 1, 2))
    images = torch.from_numpy(input_images)
    depth = model(images).detach().numpy()
    depth = np.transpose(depth[0], (1,2,0))/max_depth * 255


    rgb_frame = cv2.applyColorMap(
        depth.astype(np.uint8), cv2.COLORMAP_RAINBOW)
    print(raw_image)
    print(f"done in {(time.time() - start_time):.2f}s")
    cv2.imshow("image", raw_image)
    cv2.imshow("depth", rgb_frame)
    k = cv2.waitKey(0)
    if k == 27:
        cap.release() # release the VideoCapture object.
        break
    print("capture new frame")



 
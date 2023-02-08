import zmq
import numpy as np
import cv2
import struct

context = zmq.Context()
#socket = context.socket(zmq.SUB)
socket2 = context.socket(zmq.SUB)
#socket.connect("tcp://10.56.87.20:5556")
socket2.connect("tcp://10.56.87.20:5557")

#socket.setsockopt(zmq.SUBSCRIBE, b"image")
socket2.setsockopt(zmq.SUBSCRIBE, b"vision")

#height = 376
height = 720
#width = 672
width = 1280
channels = 3
format = np.dtype([('id_', np.uint8), ('x', np.float32), ('y', np.float32), ('z', np.float32), ('vx', np.float32), ('vy', np.float32), ('vz', np.float32)])
while True:
#    [address, img_data] = socket.recv_multipart()
    [address2, binary_message] = socket2.recv_multipart()
    num_elements = len(binary_message) // format.itemsize
    # print(len(binary_message))
    # print(len(format))
    floats_list = np.frombuffer(binary_message, dtype=format, count=num_elements)
    print(floats_list)
    print("\n")
#    img_array = np.frombuffer(img_data, dtype=np.uint8)
#//    img = img_array.reshape(height, width, channels)
    # #
#//    cv2.imshow("Image", img)
    # #
#//    if cv2.waitKey(1) & 0xFF == ord('q'):
#//        break

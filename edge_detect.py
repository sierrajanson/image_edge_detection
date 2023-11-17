import numpy as np
import cv2 
from matplotlib import pyplot as plt
from PIL import Image as im 
import os 
import time

# sucessful edge detection
def image_edge_render(path='fractal.jpg'):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def video_edge_render():
    vid = cv2.VideoCapture(0)
    image_save_file_arr = None # returns ndarray 
    while(True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            image_save_file_arr = frame
            break
    vid.release()
    #cv2.destroyAllWindows()

    #final = np.reshape(image_save_file_arr)
    image_save_file = np.ascontiguousarray(image_save_file_arr)
    final = im.fromarray(image_save_file, 'RGB')
    cv2.imwrite("captured_image.jpg",image_save_file_arr)
    time.sleep(3)
    image_grey = cv2.imread('captured_image.jpg',cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image_grey,100,200)
    plt.subplot(121),plt.imshow(edges,cmap = 'gray')
    plt.show()

if __name__ == "__main__":
    try:
        action = input("Enter 1 to run a demonstration of image edge detection using Canny\nEnter 2 to run your laptop's camera, capture a live photo, and enact edge detection on photo.\n")
        if action not in "12":
            print("wrong input provided")
            raise Exception
        if action == 1:
            image_edge_render()
        else:
            video_edge_render()

    except Exception:
        pass
    
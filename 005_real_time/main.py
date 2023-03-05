import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

WindowCapture.list_window_names()

# initialize the WindowCapture class
wincap = WindowCapture("Lost Ark Online")
# initalize the Vision class
vision_limestone = Vision("albion_limestong.jpg")

loop_time = time()
while True:
    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # cv.imshow("Computer Vision", screenshot)
    points = vision_limestone.find(screenshot, 0.5, "rectangles")

    # debug the loop rate
    print("FPS {}".format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

print("Done.")

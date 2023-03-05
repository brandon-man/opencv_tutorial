import cv2 as cv
import numpy as np


class Vision:
    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    # constructor
    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        self.needle.img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)

        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        self.method = method

    def find(self, haystack_img, threshold=0.5, debug_mode=None):
        result = cv.matchTemplate(haystack_img, self.needle_img, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        # print(locations)

        # first we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)

        # Apply group rectangles.
        # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
        # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
        # in the result. I've set eps to 0.5, which is:
        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv.groupRectangles(rectangles, 1, eps=0.5)
        # print(rectangles)

        points = []
        if len(rectangles):
            # print("Found needle.")

            line_color = (0, 255, 0)
            line_type = cv.LINE_4
            marker_color = (255, 0, 255)
            marker_type = cv.MARKER_CROSS

            # need to loop over all the locations and draw its rectangle
            for x, y, w, h in rectangles:
                # Determine the center position
                center_x = x + int(w / 2)
                center_y = y + int(h / 2)
                # save the points
                points.append((center_x, center_y))

                if debug_mode == "rectangles":
                    # determine box positions
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    # Draw a rectangle on our screenshot to highlight where we found the needle.
                    # The line color can be set as an RGB tuple
                    cv.rectangle(
                        haystack_img,
                        top_left,
                        bottom_right,
                        color=line_color,
                        lineType=line_type,
                        thickness=2,
                    )
                elif debug_mode == "points":
                    cv.drawMarker(
                        haystack_img,
                        (center_x, center_y),
                        marker_color,
                        markerType=marker_type,
                        markerSize=40,
                        thickness=2,
                    )

        if debug_mode:
            cv.imshow("Matches", haystack_img)
            # cv.waitKey()

        return points

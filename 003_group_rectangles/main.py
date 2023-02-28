import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def findClickPositions(
    needle_img_path, haystack_img_path, threshold=0.5, debug_mode=None
):
    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # There are 6 comparison methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    # You can see the differences at a glance here:
    # https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    # Note that the values are inverted for TM_SQDIFF and TM_SQDIFF_NORMED
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(haystack_img, needle_img, method)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    # print(locations)

    # first we need to create the list of [x, y, w, h] rectangles
    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
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
            cv.waitKey()

    return points


points = findClickPositions(
    "albion_turnip.jpg", "albion_farm.jpg", threshold=0.70, debug_mode="rectangles"
)
print(points)
print("Done")

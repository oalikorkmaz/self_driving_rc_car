import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def masked_image(image, show=False):  # H  S  V
    lower_thr = np.array([0, 0, 0])
    upper_thr = np.array([179, 255, 87])
    image_masked = cv2.inRange(image, lower_thr, upper_thr)
    if show:
        cv2.imshow("Masked Frame", image_masked)
    return image_masked

def canny(image, show=False):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    if show:
        cv2.imshow("Canny Filter", canny)

    return canny


def region_of_interest(image, show=False):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    cv2.rectangle(mask, (0, (height // 2) + 27), (width, height - 27), 255, -1)  # -1 -> fill
    roi_image = cv2.bitwise_and(image, mask)
    if show:
        cv2.imshow("roi mask", mask)
        cv2.imshow("roi", roi_image)
    return roi_image


def detect_lines(image):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10
    min_line_length = 20
    max_line_gap = 4

    lines = cv2.HoughLinesP(image, rho, theta, min_threshold, np.array([]), min_line_length, max_line_gap)

    return lines

def display_lines(image, lines):
    line_color = (255, 0, 0)
    line_width = 2
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), line_color, line_width)

    return image


def average_slope_intercept(image, lines):
    height = image.shape[0]
    width = image.shape[1]

    lane_lines = []

    # No lines found
    if lines is None:
        return lane_lines

    left_lane = []
    right_lane = []

    boundary = 1 / 3
    left_lane_area_width = width * (1 - boundary)
    right_lane_area_width = width * boundary

    for line in lines:
        for x1, y1, x2, y2 in line:
            # skip vertical lines as they have infinite slope
            if x1 == x2:
                continue

            # np.polyfit can be used to get slop and intercept from two points on the line
            coff = np.polyfit((x1, x2), (y1, y2), 1)
            slope = coff[0]
            intercept = coff[1]

            # note that y axis is inverted in matrix of images. 
            # so as x (width) increases, y(height) values decreases
            # this is reason why slope of right nane is positive and left name is negative

            # positive slop -> right lane marking  \
            #                                       \
            #                                        \
            #                                         \
            if slope > 0:
                # search area check
                if x1 > right_lane_area_width and x2 > right_lane_area_width:
                    right_lane.append((slope, intercept))


            # negative slop -> left lane marking  /
            #                                    /
            #                                   /
            #                                  /
            else:
                if x1 < left_lane_area_width and x2 < left_lane_area_width:
                    left_lane.append((slope, intercept))

    # averaging all the lines in each group to get a single line out of them
    left_avg = np.average(left_lane, axis=0)
    right_avg = np.average(right_lane, axis=0)

    # if got left lane, convert to point form from intercept form
    if len(left_lane) > 0:
        lane_lines.append(make_coordinates(image, left_avg))
    if len(right_lane) > 0:
        lane_lines.append((make_coordinates(image, right_avg)))

    return lane_lines


def make_coordinates(image, line):
    slop = line[0]
    intercept = line[1]
    height = image.shape[0]
    width = image.shape[1]

    #
    #
    #      left      right
    #     x1,y1      x1,y1
    #
    #
    #
    # x2,y2              x2,y2

    # y = mx + c
    # x = (y - c) / m

    y1 = int(height / 2)  # middle
    x1 = int((y1 - intercept) / slop)
    if x1 < 0:
        x1 = 0
    if x1 > width:
        x1 = width

    y2 = int(height)  # bottom
    x2 = int((y2 - intercept) / slop)
    if x2 < 0:
        x2 = 0
    if x2 > width:
        x2 = width

    return [[x1, y1, x2, y2]]

def steering_angle(image, lane, show=False):
    height = image.shape[0]
    width = image.shape[1]

    # if there is only one lane , we will set deviation to slope of the lane
    if len(lane) == 1:
        x1, y1, x2, y2 = lane[0][0]
        slope = x2 - x1
        x_deviation = slope

    # if two lines, get average of far end points
    else:
        l1x1, l1y1, l1x2, l1y2 = lane[0][0]
        l2x1, l2y1, l2x2, l2y2 = lane[1][0]
        average_point = int(l1x2 + l2x2 / 2)
        x_deviation = int(average_point) - int(width / 2)

        if show:
            steering_img = cv2.circle(image, (average_point, (int(height/2))), radius=3, color=(0, 0, 255), thickness=-1)
            steering_img = cv2.line(steering_img, (int(width / 2) , 0), (int(width / 2), height), (0, 255, 0), 1)
            cv2.imshow("Deviation", steering_img)

    line_length = int(height / 2)

    angle_to_middle_vertical_rad = math.atan(x_deviation / line_length)
    angle_to_middle_vertical_deg = int(angle_to_middle_vertical_rad * 180.0 / math.pi)
    steering = angle_to_middle_vertical_deg + 90

    return steering


def perspective_transform(image, show=False):
    top_left = [108, 100]
    bottom_left = [0, 240]
    top_right = [555, 100]
    bottom_right = [640, 248]

    cv2.circle(image, tuple(top_left), 5, (0,0,255), -1)
    cv2.circle(image, tuple(bottom_left), 5, (0,0,255), -1)
    cv2.circle(image, tuple(top_right), 5, (0,0,255), -1)
    cv2.circle(image, tuple(bottom_right), 5, (0,0,255), -1)

    pts1 = np.array([top_left, bottom_left, top_right, bottom_right], dtype=np.float32)
    pts2 = np.array([[0,0], [0,480], [640,0], [640,480]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(image, matrix, (640, 480))
    
    if show:
        cv2.imshow("Perspective", transformed_image)
    
    return transformed_image
        
sh = False
video = True


# RESİM BAŞLANGIÇ
if not video:
    image = cv2.imread("utils/img/test_image.jpg")
    image = cv2.resize(image, (640, 480))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_masked = masked_image(image_hsv, show=sh)
    #canny_image = canny(image_masked, show=sh)
    #perspective_image = perspective_transform(image_masked, show=sh)
    roi_image = region_of_interest(image_masked, show=sh)
    lines = detect_lines(roi_image)
    averaged_line = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_line)
    steering = steering_angle(image, averaged_line, show=sh)
    print("Teker Açısı: ", steering)
    
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    # plt.imshow(combo_image)
    # plt.show()

    cv2.imshow("combo_image", combo_image)
    cv2.waitKey(0)

# VİDEO BAŞLANGIÇ
else:
    cap = cv2.VideoCapture("utils/video/test_video.mp4")

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_masked = masked_image(frame_hsv, show=sh)
        #canny_frame = canny(frame_masked, show=sh)
        #frame_perspective = perspective_transform(frame_masked, show=sh)
        frame_roi = region_of_interest(frame_masked, show=sh)
        lines = detect_lines(frame_roi)
        averaged_line = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, averaged_line)
    
        steering = steering_angle(frame, averaged_line, show=False)
        print("Teker Açısı: ", steering)

        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)

        cv2.imshow("combo_frame", combo_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

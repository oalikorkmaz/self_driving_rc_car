import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def masked_image(image, show):  # H  S  V
    lower_thr = np.array([0, 0, 0])
    upper_thr = np.array([179, 255, 87])
    image_masked = cv2.inRange(image, lower_thr, upper_thr)
    if show:
        cv2.imshow("Masked Frame", image_masked)
    return image_masked

def canny(image, show):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    if show:
        cv2.imshow("Canny Filter", canny)

    return canny


def region_of_interest(image, show):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    cv2.rectangle(mask, (52, (height // 2) + 27), (590, height - 27), 255, -1)  # -1 -> fill
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

def steering_angle(image, lane, show):
    try:
        height = image.shape[0]
        width = image.shape[1]
        center_x = width // 2

        if len(lane) == 1:
            x1, y1, x2, y2 = lane[0][0]
            slope = x2 - x1
            x_deviation = slope

        elif len(lane) == 2:
            l1x1, l1y1, l1x2, l1y2 = lane[0][0]
            l2x1, l2y1, l2x2, l2y2 = lane[1][0]
            avg_x1 = int((l1x1 + l2x1) / 2)
            avg_y1 = int((l1y1 + l2y1) / 2)
            avg_x2 = int((l1x2 + l2x2) / 2)
            avg_y2 = int((l1y2 + l2y2) / 2)
            cv2.line(image, (avg_x1, avg_y1), (avg_x2, avg_y2), (0, 0, 255), 3)
            if show:
                cv2.imshow("Mid Line", image)
            
        else:
            raise ValueError("No Lane detected")

        line_length = int(height / 2)

        if line_length == 0:
            raise ZeroDivisionError("Height of the image results in line_length being zero")

        # Açı hesaplama
        dx = avg_x2 - avg_x1
        dy = avg_y2 - avg_y1
        steering = math.degrees(math.atan2(dy, dx))
        steering = int(steering)
        print(f"Çizginin X eksenine göre açısı: {steering} derece")
        return steering

    except ValueError as e:
        print(f"Error: {e}")
        return 90
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        return 90
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 90
    


def perspective_transform(image, show):
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

sh = True
video = False

# RESİM BAŞLANGIÇ
if not video:
    image = cv2.imread("utils/img/test_image.jpg")
    image = cv2.resize(image, (640, 480))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_masked = masked_image(image_hsv, show=sh)
    canny_image = canny(image_masked, show=sh)
    #perspective_image = perspective_transform(canny_image, show=sh)
    roi_image = region_of_interest(canny_image, show=sh)
    lines = detect_lines(roi_image)
    averaged_line = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_line)
    steering = steering_angle(image, averaged_line, show=False)
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

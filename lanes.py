import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def masked_image(img, show=False):  # H  S  V
    lower_thr = np.array([0, 0, 0])
    upper_thr = np.array([179, 255, 87])
    image_masked = cv2.inRange(img, lower_thr, upper_thr)
    if show:
        cv2.imshow("Masked Frame", image_masked)
    return image_masked

def average_slope_intercept(image, lines):
    width = image.shape[1]
    lane_lines = []

    if lines is None:
        return lane_lines
    
    left_fit = []
    right_fit = []
    boundary = 1 / 2
    left_lane_area_width = width * (1 - boundary)
    right_lane_area_width = width * boundary

    for line in lines:
        for x1, x2, y1, y2 in line: 
            if x1 == x2:
                continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                if x1 < left_lane_area_width and x2 < left_lane_area_width:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_lane_area_width and x2 > right_lane_area_width:
                    right_fit.append((slope, intercept))
    
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_coordinates(image, left_fit_average))
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_coordinates(image, right_fit_average))
    return lane_lines

def make_coordinates(image, line_parameters):
    slope = line_parameters[0] 
    intercept = line_parameters[1]
    height = image.shape[0]
    width = image.shape[1]

    y1 = int(height / 2)
    x1 = int((y1 - intercept) / slope)
    if x1 < 0:
        x1 = 0
    if x1 > width:
        x1 = width

    y2 = int(height)
    x2 = int((y2 - intercept) / slope)

    if x2 < 0:
        x2 = 0
    if x2 > width:
        x2 = width
    return [[x1, y1, x2, y2]]

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

def display_lines(image, lines):
    line_color = (255, 0, 0)
    line_width = 5
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), line_color, line_width)
    return image

def region_of_interest(image, show=False):
    vertices = np.array([[(18, 442), (140, 143), (480, 143), (600, 442)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    if show:
        cv2.imshow("roi mask", mask)
        cv2.imshow("roi", masked_image)
    return masked_image

def detect_lines(image):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10
    min_line_length = 20
    max_line_gap = 4
    lines = cv2.HoughLinesP(image, rho, theta, min_threshold, np.array([]), min_line_length, max_line_gap)

    return lines

def steering_angle(image, lane, show=False):
    correction_factor = 14  # Adjust this factor based on your camera setup and testing results

    if not lane or not isinstance(lane[0], list):
        print("No lane lines detected.")
        return 0 

    height = image.shape[0]
    width = image.shape[1]
    mid = int(width / 2)
    
    if len(lane) == 2:
        l1x1, l1y1, l1x2, l1y2 = lane[0][0]
        l2x1, l2y1, l2x2, l2y2 = lane[1][0]
        average_point = int((l1x2 + l2x2) / 2)
        x_deviation = int(average_point) - mid

    elif len(lane) == 1:
        x1, y1, x2, y2 = lane[0][0]
        slope = x2 - x1
        x_deviation = slope    
    else:
        print("Error: No Lane (steering_angle)")

    if show:
        steering_img = cv2.circle(image, (average_point, int(height/2)), radius=3, color=(0, 0, 255), thickness=-1)
        steering_img = cv2.line(steering_img, (int(width / 2), 0), (int(width / 2), height), (0, 255, 0), 1)
        cv2.imshow("Sapma (Deviation)", steering_img)

    line_length = int(height / 2)
    angle_to_middle_vertical_rad = math.atan(x_deviation / line_length)
    angle_to_middle_vertical_deg = int(angle_to_middle_vertical_rad * 180.0 / math.pi)
    steering_angle = angle_to_middle_vertical_deg + 90 + correction_factor

    cv2.line(image, (int(width/2), height),
             (int(x_deviation + mid), int(height/2)), (255, 255, 0), 5)

    return steering_angle

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
        
sh = True
video = True
steering_last_five = [0, 0, 0, 0, 0]

# RESİM BAŞLANGIÇ
if not video:
    image = cv2.imread("utils/img/test_image.jpg")
    lane_image = np.copy(image)
    image_masked = masked_image(lane_image, show=sh)
    #canny_image = canny(image_masked, show=sh)
    #perspective_image = perspective_transform(image_masked, show=sh)
    cropped_image = region_of_interest(image_masked, show=sh)
    lines = detect_lines(cropped_image)
    averaged_line = average_slope_intercept(lane_image, lines)
    print("LANE: ", averaged_line)
    line_image = display_lines(lane_image, averaged_line)
    steering = steering_angle(lane_image, averaged_line, show=sh)
    print("Teker Açısı: ", steering)
    
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    plt.imshow(combo_image)
    plt.show()

    if True:
        vertices = np.array([[(18, 442), (140, 143), (480, 143), (600, 442)]], dtype=np.int32)
        cv2.polylines(combo_image, vertices, isClosed=True, color=(0, 255, 0), thickness=2)
    
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
        cropped_frame = region_of_interest(frame_masked, show=sh)
        lines = detect_lines(cropped_frame)
        averaged_line = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, averaged_line)
    
        steering = steering_angle(frame, averaged_line, show=False)
        steering_last_five.insert(0, steering)
        average_steering = int(sum(steering_last_five)/len(steering_last_five))

        print("Teker Açısı: ", average_steering)
        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        if sh:
            vertices = np.array([[(18, 442), (140, 143), (480, 143), (600, 442)]], dtype=np.int32)
            cv2.polylines(combo_frame, vertices, isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("combo_frame", combo_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

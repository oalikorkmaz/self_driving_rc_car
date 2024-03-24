import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
     
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
        else:
            left_line = None
        
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
        else:
            right_line = None

        return np.array([left_line, right_line])
    else:
        return None



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    try:
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0 ,0), 15)
    
        
    except Exception as e:
        print(e)

    return line_image

def region_of_interest(image):
    vertices = np.array([[(0, 300), (1280, 300), (1280, 720), (0, 720)]], dtype=np.int32)
    # ROI için boş bir maske oluşturma
    mask = np.zeros_like(image)
    
    # ROI bölgesini beyaz renkte doldurma
    cv2.fillPoly(mask, vertices, 255)
    
    # ROI bölgesini orijinal görüntüyle maskeleme
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


# image = cv2.imread("utils/img/test_video.jpg")
# lane_image = np.copy(image)

# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
# averaged_line = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_line)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# # plt.imshow(canny_image)
# # plt.show() 

# cv2.imshow("result", combo_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("utils/video/test_video.mp4")

while(cap.isOpened()):
     _, frame = cap.read()
     canny_image = canny(frame)
     cropped_image = region_of_interest(canny_image)
     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
   
     if lines is not None:
         averaged_line = average_slope_intercept(frame, lines)
         line_image = display_lines(frame, averaged_line)
         combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
         cv2.imshow("result", combo_image)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
         time.sleep(0.1)
     else:
         print("Lines not found!")
    
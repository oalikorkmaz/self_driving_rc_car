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
    witdh = image.shape[1]
    lane_lines = []

    if lines is None:
        return lane_lines
    

    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_lane_area_width = witdh * (1 - boundary)
    right_lane_area_width = witdh * boundary
    
    
    for line in lines:
        for x1, y1, x2, y2 in line:
        
            if x1 == x2:
                continue

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            """
            y ekseninin görüntü matrisinde ters çevrildiğine dikkat edin. 
            böylece x (genişlik) arttıkça, y (yükseklik) değerleri azalır
            Bu nedenle sağ nane eğimi pozitif, sol nane eğimi ise negatiftir.
            """


            # negatif eğim -> sol şerit işaretlemesi    /
            #                                          /
            #                                         /
            #                                        /

            if slope < 0:
                if x1 < left_lane_area_width and x2 < left_lane_area_width:
                    left_fit.append((slope, intercept))

            # pozitif eğim -> sağ şerit işaretlemesi    \
            #                                            \
            #                                             \
            #                                              \

            else:
                if x1 > right_lane_area_width and x2 > right_lane_area_width:
                    right_fit.append((slope, intercept))

    # Her gruptaki tüm çizgilerin ortalamasını alarak tek bir çizgi elde etmek
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    if len(left_fit) > 0:
        lane_lines.append(make_coordinates(image, left_fit_average))

    if len(right_fit) > 0:
        lane_lines.append(make_coordinates(image, right_fit_average))
 
    
    return lane_lines

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    height = image.shape[0]
    width = image.shape[1]

    
    
    #      sol        sağ
    #     x1,y1      x1,y1
    #
    #
    #
    # x2,y2              x2,y2
    #
    # y = mx + c
    # x = (y - c) / m
    

    y1 = int(height / 2)
    x1 = int((y1 - intercept)/slope)
    if x1 < 0:
        x1 = 0
    if x1 > width:
        x1 = width

    y2 = int(height)
    x2 = int((y2 - intercept)/slope)

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
    try:
        line_color = (255, 0, 0)
        line_width = 5
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), line_color, line_width)
    except:
        print("none lines")    
    return image
    

    

def region_of_interest(image, show=False):
    # 1. Nokta: x=0, y=381 -> 2. Nokta: x=230, y=216 -> 3. Nokta: x=421, y=216 -> 4. Nokta: x=620, y=478
    vertices = np.array([[(0, 470), (0, 381), (230, 216), (421, 216), (620, 478)]], dtype=np.int32)
    # ROI için boş bir maske oluşturma
    mask = np.zeros_like(image)
    
    # ROI bölgesini beyaz renkte doldurma
    cv2.fillPoly(mask, vertices, 255)
    
    # ROI bölgesini orijinal görüntüyle maskeleme
    masked_image = cv2.bitwise_and(image, mask)
    
    if show ==True:
        cv2.imshow("roi mask", mask)
        cv2.imshow("roi", masked_image)

    return masked_image

def detect_lines(image):
    """
    HoughLinesP fonksiyonu, görüntülerdeki çizgileri bulmak için kullanılan etkili bir yöntemdir.
    Bu fonksiyon, görüntüyü öncelikle kenar görüntüsüne dönüştürür ve ardından çizgi olabilecek piksel 
    kümelerini belirlemek için bir oylama sistemi uygular. HoughLinesP fonksiyonu, çizgileri temsil etmek için kutupsal 
    koordinatları (uzaklık ve açı) kullanır. Parametreleri şunlardır:

    rho: Piksel cinsinden uzaklık çözünürlüğü (1 piksel iyi bir değerdir).
    theta: Açının radyan cinsinden çözünürlüğü (np.pi/180 genellikle 1 derecelik hassasiyet için kullanılır).
    min_threshold: Bir çizgi olarak kabul edilmesi için gereken minimum oy sayısı.
    min_line_length: Yeterince uzun olması için çizginin minimum piksel uzunluğu.
    max_line_gap: Bir çizginin aynı parça olarak kabul edilebilmesi için pikseller arasındaki maksimum boşluk.
    Fonksiyonun çıktısı, görüntüdeki tespit edilen çizgilerin uç noktalarını içeren bir NumPy dizisidir.
    """
    rho = 1
    theta = np.pi / 180
    min_threshold = 10
    min_line_length = 20
    max_line_gap = 4

    lines = cv2.HoughLinesP(image, rho, theta, min_threshold, np.array([]), min_line_length, max_line_gap)

    return lines

def steering_angle(image, lane, show=False):

    if not lane or not isinstance(lane[0], list):
        print("No lane lines detected.")
        return 0 
    
    height = image.shape[0]
    width = image.shape[1]

    if len(lane) == 1:
        x1, y1, x2, y2 = lane[0][0]
        slope = x2 - x1
        x_deviation = slope
    else:
        l1x1, l1y1, l1x2, l1y2 = lane[0][0]
        l2x1, l2y1, l2x2, l2y2 = lane[1][0]
        # Ortalama nokta hesaplama
        average_point = int((l1x2 + l2x2) / 2)
        # Yön devamı hesaplama
        x_deviation = average_point - width / 2

        if show:
            steering_img = cv2.circle(image, (average_point, int(height/2)), radius=3, color=(0, 0, 255), thickness=-1)
            steering_img = cv2.line(steering_img, (int(width / 2), 0), (int(width / 2), height), (0, 255, 0), 1)
            cv2.imshow("Deviation", steering_img)

    line_length = int(height / 2)
    angle_to_middle_vertical_rad = math.atan(x_deviation / line_length)
    angle_to_middle_vertical_deg = int(angle_to_middle_vertical_rad * 180.0 / math.pi)

    return angle_to_middle_vertical_deg

# Gruplama alanlarını görsel olarak görme [hata ayıklama için]
def lane_search_area(img, boundary = 1/2):
    height = img.shape[0] - 53
    width = img.shape[1]
    left_lane_area_width = int(width * (1 - boundary))
    right_lane_area_width = int(width * boundary)
    # left_region = np.zeros_like(img)
    # right_region = np.zeros_like(img)

    cv2.rectangle(img, (0, 0), (left_lane_area_width, height), (0, 244, 233), 5) 
    cv2.rectangle(img, (right_lane_area_width, 0), (width, height), (128, 0, 0), 5) 
    vertices = np.array([[(0, 480),(0, 381), (215, 216), (421, 216), (620, 478)]], dtype=np.int32)
    cv2.polylines(img, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("left and right region", img)


sh = False
video = True

# RESİM BAŞLANGIÇ
if video == False:
    image = cv2.imread("utils/img/test_image.jpg")
    lane_image = np.copy(image)
    image_masked = masked_image(lane_image, show=sh)
    
    canny_image = canny(image_masked, show=sh)
    cropped_image = region_of_interest(canny_image, show=sh)
    lines = detect_lines(cropped_image)
    averaged_line = average_slope_intercept(lane_image, lines)
    print("LANE : " , averaged_line)
    line_image = display_lines(lane_image, averaged_line)
    #lane_search_area(lane_image, boundary=1/3)
    steering = steering_angle(lane_image, averaged_line, show=True)
    print("Teker Açısı: ", steering)
    
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    # plt.imshow(combo_image)
    # plt.show() 

    # ROI bölgesini çizme
    if sh == True:
        vertices = np.array([[(0, 480),(0, 381), (215, 216), (421, 216), (620, 478)]], dtype=np.int32)
        cv2.polylines(combo_image, vertices, isClosed=True, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("combo_image", combo_image)
    cv2.waitKey(0)



#VİDEO BAŞLANGIÇ
else:
    cap = cv2.VideoCapture("utils/video/test_video.mp4")

    while(cap.isOpened()):
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_masked = masked_image(frame_hsv, show= sh)
        canny_frame = canny(frame_masked, show= sh)
        cropped_frame = region_of_interest(canny_frame, show=sh)
        lines = detect_lines(cropped_frame)
        averaged_line = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, averaged_line)
    
        steering = steering_angle(frame, averaged_line, show= False)
        print("Teker Açısı: ", steering)
        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow("combo_frame", combo_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
        
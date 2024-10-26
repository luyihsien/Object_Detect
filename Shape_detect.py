import cv2
import numpy as np
import math

# 讀取影像並轉換為HSV
image = cv2.imread('Normal_hearing_audiogram_TC.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 設定紅色遮罩範圍
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = red_mask1 | red_mask2  # 合併兩個紅色範圍的遮罩

# 設定藍色遮罩範圍
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 檢測紅色的圓圈
red_circles = cv2.HoughCircles(red_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=10)

# 如果找到紅色圓形，將其繪製在原圖上
if red_circles is not None:
    red_circles = np.round(red_circles[0, :]).astype("int")
    for (x, y, r) in red_circles:
        cv2.circle(image, (x, y), r, (0, 0, 0), 4)  # 用黑色框出圓形

# 使用藍色遮罩檢測叉叉，進行邊緣檢測後進行線條檢測
edges = cv2.Canny(blue_mask, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=5)

def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# 尋找相交且接近垂直的線條
if lines is not None:
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            angle1 = calculate_angle(line1)
            angle2 = calculate_angle(line2)
            
            # 檢查是否接近垂直（約90度）
            if abs(abs(angle1 - angle2) - 90) < 10:
                # 繪製叉叉的線條
                x1, y1, x2, y2 = line1[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
                
                x1, y1, x2, y2 = line2[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)

# 顯示結果
cv2.imshow("Detected Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

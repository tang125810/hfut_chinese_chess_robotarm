import math

import cv2
import numpy as np
import time


global_heart=[]
thresh_board_value=160

def find_largest_black_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_board_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect_area = 0
    max_rect_box = None
    cv2.imshow("thresh", thresh)
    # cv2.imshow("gray", gray)
    cv2.waitKey(1)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            rect_area = cv2.contourArea(approx)
            if rect_area > max_rect_area and rect_area > 3000:
                # 计算四边形的边长
                edges = [
                    np.linalg.norm(approx[0] - approx[1]),
                    np.linalg.norm(approx[1] - approx[2]),
                    np.linalg.norm(approx[2] - approx[3]),
                    np.linalg.norm(approx[3] - approx[0])
                ]
                # 判断边长是否近似相等
                max_edge = max(edges)
                min_edge = min(edges)
                if max_edge / min_edge < 1.2:  # 边长差异的容忍度
                    cv2.drawContours(image, [approx[0]], 0, (0, 255, 0), 2)
                    cv2.imshow("ssssd",image)
                    max_rect_area = rect_area
                    max_rect_box = approx.reshape(4, 2)
    return max_rect_box

def four_point_transform1(image, points, left_top_coordinate):
    src_pts = np.array(points, dtype="float32")
    maxWidth = 720
    maxHeight = 800
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 将 left_top_coordinate 转换到变换后的图像中
    transformed_left_top = cv2.perspectiveTransform(np.array([[left_top_coordinate]], dtype="float32"), M)[0][0]
    transformed_left_top = [int(transformed_left_top[0]), int(transformed_left_top[1])]
    #print("Transformed left top coordinate:", transformed_left_top)

    # 获取图像的尺寸
    (h, w) = warped.shape[:2]

    # 定义旋转中心（这里以图像中心为旋转中心）
    center = (w // 2, h // 2)
    turn_angle = 0

    if transformed_left_top[0] > 400:
        if transformed_left_top[1] < 400:
            turn_angle = 90  # 旋转角度，正数表示顺时针旋转，负数表示逆时针旋转
        if transformed_left_top[1] > 400:
            turn_angle = 180
    if transformed_left_top[0] < 400:
        if transformed_left_top[1] < 400:
            turn_angle = 0
        if transformed_left_top[1] > 400:
            turn_angle = -90

    scale = 1.0
    M_rotate = cv2.getRotationMatrix2D(center, turn_angle, scale)
    warped = cv2.warpAffine(warped, M_rotate, (w, h))

    return warped

def find_board(image):
    frame = image.copy()
    points = find_largest_black_rectangle(frame)
    if points is not None:
        cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
    #cv2.imshow("frame", frame)
    #cv2.waitKey(1)
    if points is not None:
        points = [points[1], points[0], points[3], points[2]]
        return points
    else:
        return None


def broad_coords(chess_points,left_top):
    # 重新排序 chess_points_transformed，让离 left_top_coordinate 最近的点排到第一位
    chess_points = np.array(chess_points)
    chess_point1=chess_points[0]
    chess_point2=chess_points[1]
    if (abs(chess_point1[0] - left_top[0]) + abs(chess_point1[1] - left_top[1])) > (
            abs(chess_point2[0] - left_top[0]) + abs(chess_point2[1] - left_top[1])):
        points_homogeneous = np.hstack((chess_points, np.ones((chess_points.shape[0], 1))))
        center=chess_points[4]
        # 创建旋转矩阵
        M_rotate = cv2.getRotationMatrix2D(center, -90, 1.0)

        # 应用旋转矩阵
        rotated_points_homogeneous = np.dot(M_rotate, points_homogeneous.T).T

        # 将齐次坐标转换回笛卡尔坐标
        chess_points = rotated_points_homogeneous[:, :2]
    return chess_points

def find_2_point(points):
    points = np.array(points)
    left_top=points[0]
    right_down=points[2]
    for point in points:
        if point[1] +point[0] < left_top[1] +left_top[0]:
            left_top = point
        if point[1] +point[0] > right_down[1] +right_down[0]:
            right_down = point
    return left_top, right_down

def coord_expand(points):
    points = np.array(points)

    # 计算四边形的中心点
    center = np.mean(points, axis=0)

    # 扩展比例
    k = 1.2  # 例如，向外扩展20%

    # 计算每个点相对于中心点的向量
    vectors = points - center

    # 等比例扩展向量
    expanded_vectors = vectors * k

    # 计算扩展后的点坐标
    expanded_points = center + expanded_vectors
    return expanded_points

def find_all_circle(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # 应用自适应阈值
    binary_adaptive_gaussian = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        21
    )
    # cv2.imshow("gray", gray_image)
    # cv2.imshow("binary_image", binary_image)
    cv2.imshow("binary_adaptive_gaussian", binary_adaptive_gaussian)

    # 使用霍夫圆变换检测圆形
    # 参数说明：
    # dp: 分辨率比率，dp=1时，累加器的分辨率与输入图像相同
    # minDist: 检测到的圆心之间的最小距离
    # param1: Canny边缘检测器的高阈值
    # param2: 中心检测阶段的累加器阈值
    # minRadius 和 maxRadius: 圆的最小和最大半径
    circles = cv2.HoughCircles(
        binary_adaptive_gaussian,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1= 140 ,
        param2=40,
        minRadius=10,
        maxRadius=50
    )

    circle_heart_list =[]
    circle_radius_list =[]
    # 如果检测到圆，在原图上绘制圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # 提取圆的中心和半径
            center_x, center_y, radius = circle

            # 绘制圆的轮廓
            # cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
            flag = 0
            for index,circle_heart in enumerate(circle_heart_list):

                if math.sqrt(pow((math.fsum(circle_heart[0])/len(circle_heart[0]))-center_x,2)+pow((math.fsum(circle_heart[1])/len(circle_heart[1]))-center_y,2))<10:
                    flag = 1
                    circle_heart[0].append(center_x)
                    circle_heart[1].append(center_y)
            if flag ==1:
                continue
            else:
                circle_heart_list.append([[center_x],[center_y]])
        for index, circle_heart in enumerate(circle_heart_list):
            # 绘制圆心
            cv2.circle(image, (int(math.fsum(circle_heart[0])/len(circle_heart[0])), int(math.fsum(circle_heart[1])/len(circle_heart[1]))), 28, (0,255, 0 ), 3)

    return image


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    init_points=None
    _,frame=cap.read()
    init_frame=None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if init_points is None:
            init_points = find_board(frame)
            if init_points is not None:
                init_points=coord_expand(init_points)
                init_left_top_coordinate, init_right_down_coordinate = find_2_point(init_points)

        if init_points is not None:
            init_frame = four_point_transform1(frame, init_points, init_left_top_coordinate)




        ciecles=find_all_circle(init_frame)

        cv2.imshow('frame', frame)

        cv2.imshow('Original Image', init_frame)
        cv2.imshow('Circles', ciecles)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
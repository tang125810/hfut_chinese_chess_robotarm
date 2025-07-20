import cv2
import numpy as np
import time


thresh_board_value=161
thresh_white_value=70
thresh_black_value=100

chess=[[[0,0],[200,200]],[[200,0],[400,200]],[[400,0],[600,200]],
       [[0,200],[200,400]],[[200,200],[400,400]],[[400,200],[600,400]],
       [[0,400],[200,600]],[[200,400],[400,600]],[[400,400],[600,600]]]

def coord_transfer(distance,point,center_x,center_y):
    distance_pixel_rate=100/distance
    dx=point[0]-center_x
    dy=point[1]-center_y
    DX=dx*distance_pixel_rate
    DY=dy*distance_pixel_rate


def center_image(image,point):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 计算图像的中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 在图像中心绘制十字标记
    # 设置十字的长度和颜色
    cross_size = 20  # 十字的长度
    color = (0, 255, 0)  # 绿色，格式为 BGR
    thickness = 2  # 线条粗细
    coord_transfer(distance, point, center_x, center_y)
    # 绘制水平线
    cv2.line(image, (center_x - cross_size, center_y), (center_x + cross_size, center_y), color, thickness)
    # 绘制垂直线
    cv2.line(image, (center_x, center_y - cross_size), (center_x, center_y + cross_size), color, thickness)

def pieces_classify(warped_image):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
#    gray=remove_shadows_gray_world(gray)
    #cv2.imshow('greay',gray)
    _, thresh_black = cv2.threshold(gray, thresh_black_value, 255, cv2.THRESH_BINARY_INV)
    thresh_black = cv2.GaussianBlur(thresh_black, (9,9), 2)
    gray=255-gray
    _,thresh_white = cv2.threshold(gray, thresh_white_value, 255, cv2.THRESH_BINARY_INV)
    thresh_white = cv2.GaussianBlur(thresh_white, (9, 9), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh_black = cv2.erode(thresh_black, kernel, iterations=1)
    thresh_white = cv2.erode(thresh_white, kernel, iterations=1)
    #thresh_white = cv2.dilate(thresh_white, kernel1, iterations=1)
    #thresh_white = cv2.GaussianBlur(thresh_white, (9, 9), 2)
    find_all_white_rectangle(warped_image.copy(),thresh_black,thresh_white)

def find_bigest_circle(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 1)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        max_circle = None
        max_radius = 0
        for (x, y, r) in circles:
            if r > max_radius:
                max_radius = r
                max_circle = (x, y, r)
        if max_circle is not None and r>25:
            return 1
        else:
            return 0
    else:
        return 0

def find_all_white_rectangle(result_image,thresh_black,thresh_white):
    cv2.line(result_image,(200,0),(200,600),(255,0,0),2)
    cv2.line(result_image, (400, 0), (400, 600), (255, 0, 0), 2)
    cv2.line(result_image, (0, 200), (600, 200), (255, 0, 0), 2)
    cv2.line(result_image, (0, 400), (600, 400), (255, 0, 0), 2)
    for i in range(9):
        chess_piece=chess[i]
        thresh_black_region=thresh_black[chess_piece[0][1]:chess_piece[1][1], chess_piece[0][0]:chess_piece[1][0]]
        thresh_white_region=thresh_white[chess_piece[0][1]:chess_piece[1][1], chess_piece[0][0]:chess_piece[1][0]]
        black_circle_find = find_bigest_circle(thresh_black_region)
        white_circle_find = find_bigest_circle(thresh_white_region)
        chess_color = 'None'
        mean_value=i+1
        if black_circle_find == 1:
            chess_color = 'black'
        if white_circle_find == 1:
            chess_color = 'white'
        cv2.putText(result_image, f'{chess_color} {mean_value}', (chess_piece[0][0] + 20, chess_piece[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
    cv2.imshow("result", result_image)
    cv2.waitKey(1)
    pass


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


def four_point_transform(image, points, left_top_coordinate):
    src_pts = np.array(points, dtype="float32")
    maxWidth = 600
    maxHeight = 600
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    chess_points = [
        np.array([100, 100], dtype="float32"),
        np.array([300, 100], dtype="float32"),
        np.array([500, 100], dtype="float32"),
        np.array([100, 300], dtype="float32"),
        np.array([300, 300], dtype="float32"),
        np.array([500, 300], dtype="float32"),
        np.array([100, 500], dtype="float32"),
        np.array([300, 500], dtype="float32"),
        np.array([500, 500], dtype="float32"),
    ]

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

    if transformed_left_top[0] > 300:
        if transformed_left_top[1] < 300:
            turn_angle = 90  # 旋转角度，正数表示顺时针旋转，负数表示逆时针旋转
        if transformed_left_top[1] > 300:
            turn_angle = 180
    if transformed_left_top[0] < 300:
        if transformed_left_top[1] < 300:
            turn_angle = 0
        if transformed_left_top[1] > 300:
            turn_angle = -90

    scale = 1.0
    M_rotate = cv2.getRotationMatrix2D(center, turn_angle, scale)
    warped = cv2.warpAffine(warped, M_rotate, (w, h))

    # 将 chess_points 的点坐标应用到旋转矩阵 M_rotate 中
    chess_points_rotated = []
    for point in chess_points:
        point_homogeneous = np.array([point[0], point[1], 1], dtype="float32")
        transformed_point = point_homogeneous[:2]
        chess_points_rotated.append(transformed_point)

    # 应用透视逆变换
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    chess_points_transformed = []
    for point in chess_points_rotated:
        point_homogeneous = np.array([point[0], point[1], 1], dtype="float32")
        transformed_point_homogeneous = np.dot(M_inv, point_homogeneous.T)
        transformed_point = transformed_point_homogeneous[:2] / transformed_point_homogeneous[2]
        chess_points_transformed.append(transformed_point)

    # 在原始图像上标记转换后的点
    for point in chess_points_transformed:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 255, 0), -1)  # 使用绿色标记点

    return warped , chess_points_transformed

def four_point_transform1(image, points, left_top_coordinate):
    src_pts = np.array(points, dtype="float32")
    maxWidth = 800
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

def find_nearest_point(rect_points, target_point):
    rect_points = np.array(rect_points)
    target_point = np.array(target_point)
    distances = np.linalg.norm(rect_points - target_point, axis=1)
    nearest_idx = np.argmin(distances)
    return rect_points[nearest_idx]

def get_rectangle_angle(rect_points):
    rect_points = np.array(rect_points)
    center = np.mean(rect_points, axis=0)
    vectors = rect_points - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi
    positive_angles = angles[angles >= 0]
    if len(positive_angles) > 0:
        min_positive_angle = np.min(positive_angles)
    else:
        min_positive_angle = np.min(angles) + 360
    rectangle_angle = min_positive_angle % 90
    return rectangle_angle

def get_w_coord_angle(point1,point2):
    point1=np.array(point1)
    point2=np.array(point2)
    angles = np.arctan2(point2[1]-point1[1], point2[0]-point1[0]) * 180 / np.pi
    #print("angles:", angles)
    return angles

def distance_tarnsfer(rect_points,distance_t):
    rect_points = np.array(rect_points)
    distance1 = np.sqrt((rect_points[0][0] - rect_points[1][0]) ** 2 + (rect_points[0][1] - rect_points[1][1]) ** 2)
    distance2 = np.sqrt((rect_points[1][0] - rect_points[2][0]) ** 2 + (rect_points[1][1] - rect_points[2][1]) ** 2)
    distance3 = np.sqrt((rect_points[2][0] - rect_points[3][0]) ** 2 + (rect_points[2][1] - rect_points[3][1]) ** 2)
    distance4 = np.sqrt((rect_points[0][0] - rect_points[3][0]) ** 2 + (rect_points[0][1] - rect_points[3][1]) ** 2)
    distance = np.mean([distance1, distance2, distance3, distance4])
    if distance_t is None:
        distance_t = distance
    distance = (distance+distance_t)/2
    #print(distance1,distance2,distance3,distance4,distance)
    return distance

def truth_coord(distance,chess_points,image):
    TC_rate=100/distance
    chess_points = np.array(chess_points)
    truth_value=[]
    height, width = image.shape[:2]

    # 计算图像的中心点坐标
    x = width // 2
    y = height // 2
    for point in chess_points:
        DX=(point[0]-x)*TC_rate
        DY=(point[1]-y)*TC_rate
        DX=round(DX, 2)
        DY=round(DY, 2)
        truth_value.append([DX,DY])
    return truth_value


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
    k = 1.3  # 例如，向外扩展20%

    # 计算每个点相对于中心点的向量
    vectors = points - center

    # 等比例扩展向量
    expanded_vectors = vectors * k

    # 计算扩展后的点坐标
    expanded_points = center + expanded_vectors
    return expanded_points


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    time_t = time.time()
    points = None
    init_points=None
    left_top_coordinate = None
    right_down_coordinate = None
    init_angle =None
    distance=None
    _,frame=cap.read()
    distance_plus_time=0
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


        cv2.imshow('frame', frame)

        current_time = time.time()
        if current_time - time_t >= 2 and init_frame is not None:
            points = find_board(init_frame)
            time_t = current_time
        if points is not None :
            # 确保左上角点始终是透视变换后的左上角点
            if left_top_coordinate is None and right_down_coordinate is None:
                left_top_coordinate,right_down_coordinate=find_2_point(points)
            else:
                left_top_coordinate = find_nearest_point(points, left_top_coordinate)
                right_down_coordinate = find_nearest_point(points, right_down_coordinate)

            if distance_plus_time<10:
                distance=distance_tarnsfer(points,distance)
                distance_plus_time+=1
            warped_image,chess_points = four_point_transform(init_frame, points, left_top_coordinate)
            chess_points=broad_coords(chess_points,left_top_coordinate)
            truth_points=truth_coord(distance,chess_points,init_frame)
            print(truth_points)
            print(distance)
            for i in range(0,len(chess_points)):
                chess_point=chess_points[i]
                cv2.putText(init_frame,f' {i+1}',(int(chess_point[0])+2,int(chess_point[1])+2),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pieces_classify(warped_image)
            coord_angle = get_w_coord_angle(left_top_coordinate, right_down_coordinate) -45
            center_image(init_frame, left_top_coordinate)
            cv2.putText(init_frame, f'Angle: {coord_angle:.2f} degrees', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.circle(init_frame, tuple(left_top_coordinate), 10, (0, 0, 255), 2)
            cv2.circle(init_frame, tuple(right_down_coordinate), 10, (0, 255, 255), 2)
        cv2.imshow('Original Image', init_frame)
        #cv2.imshow('warped image', warped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
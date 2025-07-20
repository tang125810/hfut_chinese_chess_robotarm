import math

import cv2
import numpy as np
import time


global_heart=[]
thresh_board_value=160

def find_chess_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_board_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)

    # 对图像进行腐蚀操作
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = 255 - thresh
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect_area = 0
    max_rect_box = None
    cv2.imshow("thresh", thresh)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(1)
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
                if max_edge / min_edge < 1.5:  # 边长差异的容忍度
                    cv2.drawContours(image, [approx[0]], 0, (0, 255, 0), 2)
                    #cv2.imshow("ssssd",image)
                    max_rect_area = rect_area
                    max_rect_box = approx.reshape(4, 2)
    return max_rect_box

def find_black_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_board_value, 255, cv2.THRESH_BINARY_INV)
    #thresh = 255 - thresh
    kernel = np.ones((3, 3), np.uint8)

    # 对图像进行腐蚀操作
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect_area = 0
    max_rect_box = None
    #cv2.imshow("thresh_balck", thresh)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(1)
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
                if max_edge / min_edge < 1.5:  # 边长差异的容忍度
                    cv2.drawContours(image, [approx[0]], 0, (0, 255, 0), 2)
                    #cv2.imshow("ssssd", image)
                    max_rect_area = rect_area
                    max_rect_box = approx.reshape(4, 2)
    return max_rect_box


def perspective_transform_image(image, points):
    src_pts = np.array(points, dtype="float32")
    maxWidth = 800
    maxHeight = 800
    dst_pts = np.array([
        [0, 0],  #左上
        [maxWidth - 1, maxHeight - 1],  #右下
        [0, maxHeight - 1],  #左下
        [maxWidth - 1, 0]  #右上
    ], dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def perspective_transform_grid_image(image, points):
    src_pts = np.array(points, dtype="float32")
    maxWidth = 640
    maxHeight = 720
    dst_pts = np.array([
        [50+0, 50+0],  #左上
        [50+maxWidth - 1, 50+maxHeight - 1],  #右下
        [50+0, 50+maxHeight - 1],  #左下
        [50+maxWidth - 1, 50+0]  #右上
    ], dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (100+maxWidth,100+ maxHeight))
    return warped

def inv_perspective_transform_points(points,target_point):
    dst_pts = np.array(points, dtype="float32")
    maxWidth = 800
    maxHeight = 800
    src_pts = np.array([
        [0, 0],  #左上
        [maxWidth - 1, maxHeight - 1],  #右下
        [0, maxHeight - 1],  #左下
        [maxWidth - 1, 0]  #右上
    ], dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 定义一组要变换的点
    points_to_transform = np.float32(target_point).reshape(-1, 1, 2)

    # 应用透视变换
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)
    if transformed_points.any():
        return [[int(transformed_points[i][0][0]),int(transformed_points[i][0][1])] for i in range(transformed_points.shape[0])]
    return None  #将变换点返回

def find_board(image):
    frame = image.copy()
    points = find_chess_board(frame)
    if points is not None:
        cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
    #cv2.imshow("frame", frame)
    #cv2.waitKey(1)
    if points is not None:
        points = [points[1], points[0], points[3], points[2]]
        return points
    else:
        return None

def find_black_board_points(image):
    frame = image.copy()
    points = find_black_board(frame)
    if points is not None:
        # cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
        # cv2.imshow("find_black_board_points",frame)
        return sort_4_points(points)
    return None



def sort_4_points(points):
    points = np.array(points)

    up_min =100000
    up_max = 0
    down_min = 100000
    down_max = 0


    up_min_index = 0
    down_min_index = 0
    up_max_index = 0
    down_max_index = 0

    for index,point in enumerate(points):
        up_length = math.sqrt(pow(point[0] - 0, 2) + pow(point[1], 2))
        down_length = math.sqrt(pow(point[0]-0,2)+pow(point[1]-480,2))
        if up_length<up_min:
            up_min = up_length
            up_min_index = index
        if up_length>up_max:
            up_max = up_length
            up_max_index = index

        if down_length<down_min:
            down_min = down_length
            down_min_index = index
        if down_length>down_max:
            down_max = down_length
            down_max_index = index

    #left_top, right_bottom , left_bottom, right_top
    return points[up_min_index],points[up_max_index],points[down_min_index],points[down_max_index]

def coord_expand(points):
    points = np.array(points)

    # 计算四边形的中心点
    center = np.mean(points, axis=0)

    # 扩展比例
    k = 1  # 例如，向外扩展20%

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
    kernel = np.ones((3,3), np.uint8)

    # 对图像进行腐蚀操作
    binary_adaptive_gaussian = cv2.erode(binary_adaptive_gaussian, kernel, iterations=1)
    #cv2.imshow("binary_adaptive_gaussian", binary_adaptive_gaussian)

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

def draw_corner_points(points, image):
    points = np.array(points)
    for i  in range(len(points)):
        cv2.circle(image, (int(points[i][0]), int(points[i][1])), 5,
               [255, 0, int(255/(i+1))],
               3)

def on_trackbar(val):
    # 获取当前滑动条的值
    global  thresh_board_value
    thresh_board_value = val



if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    init_points = None  #不主动更新的 棋盘布四角位置 排序后的四个角点 左上角 右下角 左下角 右上角
    _, frame = cap.read()
    init_frame = None
    init_pointts_t = None  #棋盘布四角位置 实时更新  排序后的四个角点 左上角 右下角 左下角 右上角

    state_1_finished = False
    state_2_finished = False

    inner_point_updated = False
    inner_point_in_frame =[]

    # global_waped_board=None
    global_inner_point=None
    cv2.namedWindow("frame")
    # 创建滑动条
    cv2.createTrackbar("Threshold", "frame", 0, 255, on_trackbar)
    while True:
        try:
            print(thresh_board_value)
            # print(state_1_finished)
            # print(init_points)
            save_image_list = []
            output_list = []
            frame_num_list = []
            ret, frame = cap.read()
            final_frame = frame.copy()
            if not ret:
                break

            ''' 寻找棋盘的四个角点 然后排序 画圈'''
            point = find_board(frame)
            if point:
                init_pointts_t = point
                init_pointts_t = coord_expand(init_pointts_t) #将棋盘布坐标 按比例 扩展
                init_pointts_t = sort_4_points(init_pointts_t)  # 获取排序后的四个角点 左上角 右下角 左下角 右上角
                draw_corner_points(init_pointts_t, frame)  #绘制棋盘布的点





            if state_1_finished == True :
                try:
                    # pass
                    #将棋盘透视变换
                    waped_board = perspective_transform_image(frame,init_points)
                    # global_waped_board = waped_board
                    '''  '''

                    #在棋盘布的透视变换后图像里面 寻找 棋盘网格
                    inner_grid_points = find_black_board_points(waped_board)
                    '''  '''


                    # if 1:  #绘制图像
                    #     for point in inner_grid_points:
                    #         # 确保点的坐标是整数
                    #         x, y = int(point[0]), int(point[1])
                    #         cv2.circle(waped_board, (x, y), 7, (0,0,255), 2)
                    #     global_waped_board = waped_board
                        # cv2.imshow("waped_board", waped_board)


                    #使用逆变换 获取棋盘网格在原始frame里的坐标
                    if inner_point_updated == False:
                        inner_point_in_frame = inv_perspective_transform_points(init_points,inner_grid_points)
                        '''  '''


                        inner_point_updated = True
                    else:
                        inner_point_in_frame = inner_point_in_frame
                        inner_point_updated=False
                    inner_point = []
                    if inner_point_in_frame and 1:  # 绘制在原始 frame中的 棋盘网格点
                        # for item in inner_point_in_frame:
                        #     print(item[0])
                        for point in inner_point_in_frame:
                            # point=point[0]
                            # 确保点的坐标是整数
                            x, y = int(point[0]), int(point[1])
                            inner_point.append([x,y])
                            cv2.circle(frame, (x, y), 6, (0,0,255), 2)
                    #以棋盘网格为 锚点 透视变换 原始frame 获得 最后效果图
                    if len(inner_point) == 4:
                        global_inner_point = inner_point
                        '''  '''
                        # final_grid = perspective_transform_grid_image(final_frame,inner_point)
                        # cv2.imshow("final_grid", final_grid)
                    state_1_finished = False
                except:
                    state_1_finished = False
                    # global_waped_board = None
                    global_inner_point = None


            if  global_inner_point:

                final_grid = perspective_transform_grid_image(final_frame, global_inner_point)

                for point in global_inner_point:
                    cv2.circle(frame, (point[0],point[1]), 6, (0, 0, 255), 2)

                cv2.imshow("final_grid", final_grid)
                # cv2.imshow("global_waped_board", global_waped_board)
            else:
                pass

            cv2.imshow('frame', frame)




            user_input = cv2.waitKey(1)

            # 初始调用一次回调函数
           # on_trackbar(1)
            if user_input == ord(' '):
                # 按下q 表示跟新一次 静态棋盘布点 同时切换一次 显示模式
                if state_1_finished == True:
                    state_1_finished = False
                    try:
                        cv2.destroyWindow("waped_board")
                        cv2.destroyWindow("final_grid")
                    except:
                        pass
                elif state_1_finished == False:
                    state_1_finished = True
                    if init_pointts_t:
                        init_points = init_pointts_t
                    else:
                        state_1_finished = False



        except:
            pass


    cap.release()
    cv2.destroyAllWindows()
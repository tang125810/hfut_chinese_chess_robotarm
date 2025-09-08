import math
import json
import threading
import time

import cv2

from openvino_pretect import*
import socket
from game_control import Game_board
from serial_utils import ArmCommThread
from socket_to_chinesechess2 import SocketThread

arm_comm_serial = ArmCommThread(port='com17')
threading.Thread(target=arm_comm_serial.run).start()

left_black=115
right_black=525
top_black=61
bottom_black=471
send_data=None

model_xml = "best11.xml"
model_bin = "best11.bin"

ov_inference = OpenVINOInference(model_xml, model_bin, device_name='GPU')
ov_inference.print_available_devices()


game_board =  Game_board("all_chess_config.json")
[print(item) for item in game_board.get_classify_grid_board()]

chess_image_lists = [None for i in range(32)]
chess_position = [[14 for _ in range(9)] for _ in range(10)]  # 类型位置棋盘
old_chess_position = [[14 for _ in range(9)] for _ in range(10)]
old_occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)] # 占位位置棋盘
physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 物理位置棋盘
old_physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 像素位置棋盘
alive_chess_cout =32


history_chess_position= []
chess_img_array = np.zeros((32, 60, 60, 3), dtype=np.uint8)
allow_change = True
global_heart=[]
thresh_board_value=124

if_para2_to34=False

circle_param1 = 106
circle_param2 = 38
circle_radius1 = 6
circle_radius2 = 48

save_inner_grid_points=[]

start_chess_point=None
end_chess_point=None


server_address = ('localhost', 12345)
client_socket = SocketThread(server_address)
client_socket.start()


def find_chess_board(image):
    global left_black
    global right_black
    global top_black
    global bottom_black
    global thresh_x
    global thresh_y
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_board_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)

    # 对图像进行腐蚀操作
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = 255 - thresh

    thresh_width=thresh.shape[1]
    thresh_height=thresh.shape[0]
    thresh[0:top_black,:]=0
    thresh[bottom_black:thresh_height,:] = 0
    thresh[:, 0:left_black] = 0
    thresh[:, right_black:thresh_width] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect_area = 0
    max_rect_box = None
    cv2.imshow("thresh", thresh)
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
                    max_rect_area = rect_area
                    max_rect_box = approx.reshape(4, 2)
    return max_rect_box

def find_Black_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_board_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    # 对图像进行腐蚀操作
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect_area = 0
    max_rect_box = None
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
    if points is not None:
        points = [points[1], points[0], points[3], points[2]]
        return points
    else:
        return None

def find_Black_board_points(image):
    frame = image.copy()
    points = find_Black_board(frame)
    if points is not None:
        # cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
        # cv2.imshow("find_Black_board_points",frame)
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

    global circle_param1
    global circle_param2
    global circle_radius1
    global circle_radius2
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # 应用自适应阈值
    binary_adaptive_gaussian = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        21
    )

    kernel1 = np.ones((3,3), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    # 对图像进行腐蚀操作
    binary_adaptive_gaussian = cv2.dilate(binary_adaptive_gaussian, kernel1,iterations=1)  # 原来是 iterations=2，现在改为 iterations=1
    #binary_adaptive_gaussian = cv2.erode(binary_adaptive_gaussian, kernel2, iterations=1)
    binary_adaptive_gaussian = cv2.erode(binary_adaptive_gaussian, kernel2, iterations=1)
    binary_adaptive_gaussian = cv2.dilate(binary_adaptive_gaussian, kernel1,iterations=1)  # 原来是 iterations=2，现在改为 iterations=1
    binary_adaptive_gaussian = cv2.erode(binary_adaptive_gaussian, kernel2, iterations=1)
    binary_adaptive_gaussian = cv2.erode(binary_adaptive_gaussian, kernel2, iterations=1)
    binary_adaptive_gaussian = cv2.GaussianBlur(binary_adaptive_gaussian, (15, 15), 0)


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
        param1= circle_param1 ,
        param2=circle_param2,
        minRadius=circle_radius1,
        maxRadius=circle_radius2
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


    return image,circle_heart_list
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

def on_trackbar_p1(val):
    # 获取当前滑动条的值
    global  circle_param1
    circle_param1 = val

def on_trackbar_p2(val):
    # 获取当前滑动条的值
    global  circle_param2
    circle_param2 = val

def on_trackbar_r1(val):
    # 获取当前滑动条的值
    global  circle_radius1
    circle_radius1 = val

def on_trackbar_r2(val):
    # 获取当前滑动条的值
    global  circle_radius2
    circle_radius2 = val

def on_trackbar_x1(val):
    # 获取当前滑动条的值
    global  left_black
    left_black = val

def on_trackbar_x2(val):
    # 获取当前滑动条的值
    global  right_black
    right_black = val

def on_trackbar_y1(val):
    # 获取当前滑动条的值
    global  top_black
    top_black = val

def on_trackbar_y2(val):
    # 获取当前滑动条的值
    global  bottom_black
    bottom_black = val

def start_end_chess(last_chess_board,scend_last_chess_board):
    last_chess_board_t=np.array(last_chess_board)
    scend_last_chess_board_t=np.array(scend_last_chess_board)
    detect_board=last_chess_board_t-scend_last_chess_board_t
    indices1 = np.nonzero(detect_board)
    point1=[indices1[0][0],indices1[1][0]]
    point2=[indices1[0][1],indices1[1][1]]
    if last_chess_board_t[point1[0],point1[1]]==14:
        print("起点：",point1)
        print("落点：",point2)
        return point1,point2
    elif last_chess_board_t[point2[0],point2[1]]==14:
        print("起点：", point2)
        print("落点：", point1)
        return point2,point1
    else:
        return None,None





def  Chess_Position_Update(circle_heart_list):

    global chess_position
    global  chess_img_array
    global old_chess_position
    global old_occupy_chess_position
    global start_chess_point
    global end_chess_point
    global old_occupy_chess_position
    global physical_positiopn_chess_position
    global old_physical_positiopn_chess_position
    alive_chess_cout = 0

    result = ov_inference.infer(chess_img_array)
    for i in range(len(circle_heart_list)):
        chess_classify_id = np.argmax(result[i])
        chess_classify_id = ov_inference.trans_dict[chess_classify_id]
        circle_heart=circle_heart_list[i]
        x_coordinate = int(round((circle_heart[0][0]-50)/80))
        y_coordinate = int(round((circle_heart[1][0]-50)/80))
        chess_position[y_coordinate][x_coordinate] = chess_classify_id
        physical_positiopn_chess_position[y_coordinate][x_coordinate] = [circle_heart[0][0],circle_heart[1][0]]

    old_physical_positiopn_chess_position = physical_positiopn_chess_position


    old_chess_position=chess_position
    occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)]  # x占据位置棋盘
    for y in range(10):
        for x in range(9):
            if(old_chess_position[y][x] <14):
                occupy_chess_position[y][x] = 1
                alive_chess_cout+=1
            else:
                occupy_chess_position[y][x] = 0
    old_occupy_chess_position = occupy_chess_position

    if game_board.makesure_input(old_chess_position,old_occupy_chess_position)>0:
        if game_board.makesure_input(old_chess_position,old_occupy_chess_position)==3:
            print(f"----- 棋盘没有变化 \t 全局步数 {game_board.global_time_label} -----")
            chess_position = [[14 for _ in range(9)] for _ in range(10)]  # 类型位置棋盘
            occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)]  # x占据位置棋盘
            physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 像素位置棋盘
            return -1
        else:
            try:
                ret,move_recode = game_board.get_move_recode(old_chess_position,old_occupy_chess_position,physical_positiopn_chess_position)
                print(f"------移动记录 move_recode log \t  全局步数： {move_recode['global_time_label']} -----")
                print(f" 棋子颜色： {move_recode['color']} \t 棋子兵种：{move_recode['color']}_{move_recode['name_CN']} ")
                print(f"源位置：{move_recode['from_abstruct']} ({move_recode['from_physical'][0],move_recode['from_physical'][1],}) \t 目的位置：{move_recode['to_abstruct']} ({move_recode['to_physical'][0],move_recode['to_physical'][1],})")
                try:
                    start_chess_point = move_recode['from_abstruct']
                    end_chess_point = move_recode['to_abstruct']
                except:
                    pass

                if move_recode['eat']:
                    print(f"被吃： {move_recode['dead_man_color']}_{move_recode['dead_man_name_CN']} 位置：{move_recode['dead_man_place_abstruct']} ({move_recode['dead_man_place_physical']})")
                else:
                    print(f"没有吃子")
                # print(game_board.move_recode_list[-1])
                # print("移动记录长度",len(game_board.move_recode_list))
                res = game_board.fresh(occupy_chess_position, old_chess_position, physical_positiopn_chess_position,move_recode)
                if res and game_board.move_recode_list[-1]['eat']== True:  #跟新吃子
                    alive_chess_cout -= 1
                chess_position = [[14 for _ in range(9)] for _ in range(10)]  # 类型位置棋盘
                occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)]  # x占据位置棋盘
                physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 像素位置棋盘
                return 10
            except :
                print(f"----- 移动记录更新 与 棋盘更新出错 全局步数 {game_board.global_time_label} -----")
                print(f"----- 错误记录 在game_log.json中 全局步数 {game_board.global_time_label} -----")
                print(f"物理位置")
                [print(item) for item in physical_positiopn_chess_position]
                data={
                    "chess_position":chess_position,
                    "occupy_chess_position":occupy_chess_position,
                    "physical_positiopn_chess_position":physical_positiopn_chess_position,
                    "game_board_index_grid_board":game_board.get_index_grid_board(),
                    "game_board_occupy_grid_board":game_board.get_occupy_grid_board(),
                    "game_board_classify_grid_board":game_board.get_classify_grid_board(),
                    "game_board_physical_position_grid_board":game_board.get_physical_position_grid_board()
                }
                # 将更新后的数据写回到 JSON 文件
                with open("game_log.json", 'w') as file:
                    json.dump(data, file, indent=4)  # 使用缩进格式化输出

                chess_position = [[14 for _ in range(9)] for _ in range(10)]  # 类型位置棋盘
                occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)]  # x占据位置棋盘
                physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 像素位置棋盘
                return -1

    else:
        print(f"----- 检测检测到 视觉视觉与棋盘不匹配 全局步数 {game_board.global_time_label} -----")
        chess_position = [[14 for _ in range(9)] for _ in range(10)]  # 类型位置棋盘
        occupy_chess_position = [[0 for _ in range(9)] for _ in range(10)]  # x占据位置棋盘
        physical_positiopn_chess_position = [[None for _ in range(9)] for _ in range(10)]  # 像素位置棋盘
        return -1


def serial_send(data):
    arm_comm_serial.if_open_send = True
    if data:
        arm_comm_serial.send(data)  # 自定义心跳帧
    else:
        pass
        # arm_comm_serial.send(send_data)  # 自定义心跳帧
    while 1:
        time.sleep(0.2)
        if arm_comm_serial.if_reached:
            break


if __name__ == '__main__':
    #set_camera_resolution(0, 1920, 1080)  # 设置摄像头分辨率为 1920x1080

    cap = cv2.VideoCapture(1)
    # 设置摄像头分辨率
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    init_points = None  #不主动更新的 棋盘布四角位置 排序后的四个角点 左上角 右下角 左下角 右上角
    _, frame = cap.read()
    init_frame = None
    init_pointts_t = None  #棋盘布四角位置 实时更新  排序后的四个角点 左上角 右下角 左下角 右上角

    state_1_finished = False
    state_2_finished = False

    read_inner_grid_points=False

    inner_point_updated = False
    inner_point_in_frame =[]
    circle_heart_list=None
    final_grid=None

    global_inner_point=None
    cv2.namedWindow("frame")
    cv2.namedWindow("Val")
    cv2.resizeWindow("Val", 800, 300)
    # 创建滑动条
    cv2.createTrackbar("Threshold", "frame", thresh_board_value, 255, on_trackbar)
    cv2.createTrackbar("C_p_1", "Val", circle_param1, 200, on_trackbar_p1)
    cv2.createTrackbar("C_p_2", "Val", circle_param2, 200, on_trackbar_p2)
    cv2.createTrackbar("C_r_1", "Val", circle_radius1, 70, on_trackbar_r1)
    cv2.createTrackbar("C_r_2", "Val", circle_radius2, 70, on_trackbar_r2)
    cv2.createTrackbar("left_black", "Val", left_black, 640, on_trackbar_x1)
    cv2.createTrackbar("top_black", "Val", top_black, 480, on_trackbar_y1)
    cv2.createTrackbar("right_black", "Val", right_black, 640, on_trackbar_x2)
    cv2.createTrackbar("bottom_black", "Val", bottom_black, 480, on_trackbar_y2)



    try:
        while True:
            try:
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
                        #将棋盘透视变换
                        waped_board = perspective_transform_image(frame,init_points)
                        if read_inner_grid_points == True:
                            try:
                                with open('points.json', 'r') as json_file:
                                    save_inner_grid_points = json.load(json_file)
                                #save_inner_grid_points=np.array(save_inner_grid_points)
                                inner_grid_points = sort_4_points(save_inner_grid_points)
                                print('读取save_inner_grid_points')
                                read_inner_grid_points = False
                            except:
                                read_inner_grid_points = False
                                print("读取棋盘格点失败")
                                pass

                        else:
                            # 在棋盘布的透视变换后图像里面 寻找 棋盘网格
                            inner_grid_points = find_Black_board_points(waped_board)
                            read_inner_grid_points = False


                        #使用逆变换 获取棋盘网格在原始frame里的坐标
                        if inner_point_updated == False:
                            inner_point_in_frame = inv_perspective_transform_points(init_points,inner_grid_points)
                            inner_point_updated = True
                        else:
                            inner_point_in_frame = inner_point_in_frame
                            inner_point_updated=False
                        inner_point = []
                        if inner_point_in_frame and 1:  # 绘制在原始 frame中的 棋盘网格点
                            for point in inner_point_in_frame:
                                # 确保点的坐标是整数
                                x, y = int(point[0]), int(point[1])
                                inner_point.append([x,y])
                                cv2.circle(frame, (x, y), 6, (0,0,255), 2)
                        #以棋盘网格为 锚点 透视变换 原始frame 获得 最后效果图
                        if len(inner_point) == 4:
                            global_inner_point = inner_point

                        state_1_finished = False
                    except:
                        state_1_finished = False
                        # global_waped_board = None
                        global_inner_point = None


                if  global_inner_point:

                    final_grid = perspective_transform_grid_image(final_frame, global_inner_point)
                    final_grid_copy=cv2.cvtColor(final_grid.copy(), cv2.COLOR_BGR2RGB)

                    for point in global_inner_point:
                        cv2.circle(frame, (point[0],point[1]), 6, (0, 0, 255), 2)
                    circles_image,circle_heart_list=find_all_circle(final_grid)
                    for i in range(len(circle_heart_list)):
                        circle_heart=circle_heart_list[i]
                        y1=circle_heart[1][0]-30
                        y2=circle_heart[1][0]+30
                        x1=circle_heart[0][0]-30
                        x2=circle_heart[0][0]+30

                        chess_img_array[i]=final_grid_copy[y1:y2,x1:x2]
                    cv2.imshow("final_grid", final_grid)
                else:
                    pass
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]

                cv2.line(frame, (0,top_black), (frame_width,top_black), (0,0,255), 1)
                cv2.line(frame, ( 0,bottom_black), (frame_width,bottom_black), (0,0,255), 1)
                cv2.line(frame, (left_black,0  ), (left_black,frame_height ), (0,0,255), 1)
                cv2.line(frame, (right_black,0   ), (right_black, frame_height), (0,0,255), 1)

                cv2.imshow('frame', frame)


                user_input = cv2.waitKey(1)

                # 初始调用一次回调函数
               # on_trackbar(1)
                if user_input == ord('r'):
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

                elif user_input == ord('g'):
                    try:
                        if circle_heart_list:
                            ret = Chess_Position_Update(circle_heart_list)
                    except:
                        print("检测失败")
                        pass

                elif user_input == ord('h'):
                    try:

                        client_socket.start_point=start_chess_point
                        client_socket.end_point=end_chess_point
                        client_socket.send_user_move = True
                        #client_socket.get_best_move(start_chess_point, end_chess_point)
                        while True:
                            if client_socket.get_best_move_compelete:
                                best_move=client_socket.bestmove
                                client_socket.get_best_move_compelete=False
                                break
                        #position1 = game_board.move_recode_list[-1]["from_physical"]
                        position1=old_physical_positiopn_chess_position[best_move[1]][best_move[0]]
                        position2 = [best_move[2] * 80, best_move[3] * 80]
                        position1_physical_mm = [125 - (position1[0] - 50) * 31.25 / 80,
                                                 (position1[1] - 50) * 31.25 / 80]
                        position2_physical_mm = [125 - best_move[2] * 31.25, best_move[3] * 31.25]
                        # 判断落点是否有棋子,用finish_id来表示发送数据的最后一位
                        class_of_down=old_chess_position[best_move[3]][best_move[2]]
                        if class_of_down==14:
                            finish_id=-1
                        elif 6<class_of_down<14:
                            finish_id=1
                        elif class_of_down<7:
                            finish_id=0
                        print(
                            f'/mv{int(round(position1_physical_mm[0]))},{int(round(position1_physical_mm[1]))},{int(round(position2_physical_mm[0]))},{int(round(position2_physical_mm[1]))},{int(finish_id)}')
                        print("best_move", best_move)
                        str_output = f'/mv{int(round(position1_physical_mm[0]))},{int(round(position1_physical_mm[1]))},{int(round(position2_physical_mm[0]))},{int(round(position2_physical_mm[1]))},{int(finish_id)}'
                        send_data = f'{str_output}\n'
                        serial_send(send_data)
                    except:
                        print(" 通讯中断")
                        pass

                elif user_input == ord('s'):
                    try:
                        save_inner_grid_points=inner_grid_points
                        save_inner_grid_points=np.array(save_inner_grid_points,dtype=np.int32)
                        save_inner_grid_points_list = [[int(coord) for coord in point] for point in save_inner_grid_points]
                        with open('points.json', 'w') as json_file:
                            json.dump(save_inner_grid_points_list, json_file, indent=4)
                        print('保存inner_grid_points')
                    except:
                        print("保存棋盘格点失败")
                        pass

                elif user_input == ord('d'):
                    try:

                        if read_inner_grid_points == False:
                            state_1_finished=True
                            read_inner_grid_points = True
                            inner_point_updated=False

                    except:

                        pass

                elif user_input == ord('c'):
                    try:
                        str_output=f'/cl'
                        send_data = f'{str_output}\n'
                        serial_send(send_data)
                    except:
                        print("clean下位机失败")
                        pass

                # elif user_input == ord('h'):
                #     try:
                #
                #         best_move=client_socket.get_best_move(start_chess_point,end_chess_point)
                #         position1 = old_physical_positiopn_chess_position[best_move[1]][best_move[0]]
                #         position2 = [best_move[2]*80,best_move[3]*80]
                #         position1_physical_mm = [125-(position1[0]-50)*31.25/80,(position1[1]-50)*31.25/80]
                #         position2_physical_mm = [125-best_move[2]*31.25,best_move[3]*31.25]
                #         print(f'/mv{int(round(position1_physical_mm[0]))},{int(round(position1_physical_mm[1]))},{int(round(position2_physical_mm[0]))},{int(round(position2_physical_mm[1]))},{int(best_move[-1])}')
                #         print("best_move",best_move)
                #
                #     except:
                #
                #         pass


            except:
                pass
    finally:
        client_socket.stop()
        cap.release()
        cv2.destroyAllWindows()
import socket
import threading
import cv2
import time


# ----------------------------------------------------------
# Socket 通信线程
# ----------------------------------------------------------
class SocketThread(threading.Thread):
    def __init__(self, server_address):
        super().__init__(daemon=True)
        self.server_address = server_address
        self.socket = None
        self.running = False
        self.if_input_XY = False
        self.if_get_best_move = False
        self.get_best_move_compelete = False
        self.start_point = []
        self.end_point = []
        self.send_user_move = False
        self.bestmove = []
        self.reconnect_delay = 5  # 重连延迟时间（秒）
        self.heartbeat_interval = 30  # 心跳间隔时间（秒）
        self.last_heartbeat_time = time.time()

    def run(self):
        self.running = True
        self.connect_to_server()

        while self.running:
            try:
                if self.send_user_move:
                    self.send_user_move = False
                    self.get_best_move(self.start_point, self.end_point)
                else:
                    time.sleep(0.1)

                # 每 30 秒发送一次心跳
                if time.time() - self.last_heartbeat_time > self.heartbeat_interval:
                    self.send("heartbeat")
                    self.last_heartbeat_time = time.time()

            except Exception as e:
                print(f"Exception in run: {e}")
                self.reconnect()

        if self.socket:
            self.socket.close()
        print("Socket thread stopped")

    def connect_to_server(self):
        while self.running:
            try:
                if not self.socket:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect(self.server_address)
                print("Connected to server")
                break
            except ConnectionRefusedError:
                print("Waiting for connection...")
                time.sleep(self.reconnect_delay)
            except Exception as e:
                print(f"Error connecting to server: {e}")
                self.reconnect()

    def reconnect(self):
        print("Attempting to reconnect...")
        if self.socket:
            self.socket.close()
            self.socket = None
        time.sleep(self.reconnect_delay)
        self.connect_to_server()

    def receive_message(self):
        try:
            # 接收服务器发送的消息
            message = self.socket.recv(1024)  # 1024 是缓冲区大小
            if message:
                print(f"Received message from server: {message.decode('utf-8').replace('102finish', '')}")
                return message.decode('utf-8')
            else:
                print("No message received from server.")
                return None
        except Exception as e:
            print(f"Error receiving message: {e}")
            self.reconnect()
            return None

    def get_best_move(self, start_chess_point, end_chess_point):
        try:
            # 发送数据并检测回发数据
            def send_and_check(point):
                message = f"{point[0]},{point[1]}"
                if self.send_click_point(point[0], point[1]):
                    time.sleep(0.7)
                    received = self.receive_message()
                    if received and len(received) == 7:
                        best_move = []
                        for i in range(7):
                            best_move.append(int(received[i]))
                        self.bestmove = [best_move[3], best_move[4], best_move[5], best_move[6]]
                        return True
                    elif received == f"{point[0]}":
                        print(f"Received confirmation for point {point}")
                        return True
                    else:
                        print(f"Received data does not match: {received} != {message}")
                return False

            # 发送 (99, 99)
            for i in range(3):
                if send_and_check((99, 99)):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            # 发送 start_chess_point
            for i in range(3):
                if send_and_check(start_chess_point):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            # 发送 end_chess_point
            for i in range(3):
                if send_and_check(end_chess_point):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            # 发送 (100, 100)
            for i in range(3):
                if send_and_check((100, 100)):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            # 发送 (101, 101)
            for i in range(3):
                if send_and_check((101, 101)):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            for i in range(3):
                if send_and_check((103, 103)):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            # 发送 (102, 102)
            for i in range(3):
                if send_and_check((102, 102)):
                    break
                elif i == 2:
                    print("发送异常")
                    return 0
                time.sleep(3)

            self.get_best_move_compelete = True

        except Exception as e:
            print(f"Exception in get_best_move: {e}")
            self.reconnect()
            return None

    def send_click_point(self, x, y, flag=False):
        message = f"{x},{y}".encode('utf-8')
        try:
            self.socket.sendall(message)
            if flag:
                print(f"Sent click point: ({x}, {y})")
            return True
        except Exception as e:
            print(f"Error sending click point: {e}")
            self.reconnect()
            return False

    def send(self, message):
        if self.running:
            try:
                self.socket.sendall(message.encode())
            except Exception as e:
                print("Error sending message:", e)
                self.reconnect()

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        print("Socket thread stopped")


# ----------------------------------------------------------
# 主程序
# ----------------------------------------------------------
def main():
    # 创建 Socket 线程
    server_address = ('localhost', 12345)
    socket_thread = SocketThread(server_address)
    socket_thread.start()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Press 'q' to exit")
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # 显示视频帧
        cv2.imshow("Video Stream", frame)

        # 按下 'q' 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 按下 's' 键发送消息到服务器
        if key == ord('s'):
            socket_thread.start_point = [0, 6]
            socket_thread.end_point = [0, 5]
            socket_thread.send_user_move = True
        if socket_thread.get_best_move_compelete:
            print("Best move received:", socket_thread.bestmove)
            socket_thread.get_best_move_compelete = False

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    socket_thread.stop()
    print("Program exited")


if __name__ == "__main__":
    main()
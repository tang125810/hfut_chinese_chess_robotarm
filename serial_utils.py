import cv2
import serial
import threading
import queue
import time
import struct


# ----------------------------------------------------------
# 串口通信线程
# ----------------------------------------------------------
class ArmCommThread(threading.Thread):
    def __init__(self,
                 port: str = 'com12',
                 baud: int = 115200,
                 tx_qsize: int = 10,
                 rx_qsize: int = 100):
        super().__init__(daemon=True)
        self.ser = None
        self.port = port
        self.baud = baud
        self._stop_event = threading.Event()

        # 线程安全的队列
        self.tx_q = queue.Queue(maxsize=tx_qsize)  # 主线程 -> 串口
        self.rx_q = queue.Queue(maxsize=rx_qsize)  # 串口 -> 主线程

        self.if_open_send = False
        self.if_reached = False
        self.receive_data = []
        self.if_change_send_data = False
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            print(f"[ArmComm] 串口已连接 {self.port}")
        except Exception as e:
            print(f"[ArmComm] 串口连接失败: {e}")
            self.ser = None

    def send(self, data: bytes) -> bool:
        """非阻塞方式把数据塞进发送队列"""
        if self.ser and self.ser.is_open:
            try:
                payload = data.encode('ascii') if isinstance(data, str) else data
                self.tx_q.put_nowait(payload)
                return True
            except queue.Full:
                print("[ArmComm] 发送队列已满")
        return False

    def get_latest_rx(self, block=False, timeout=None):
        """取一条最新收到的数据"""
        try:
            receive_data = self.rx_q.get(block=block, timeout=timeout)
            print("receive_data", receive_data)
            return receive_data
        except queue.Empty:
            return None

    def run(self):
        frame_buf = b''
        in_frame = False

        while not self._stop_event.is_set():
            if not self.ser or not self.ser.is_open:
                time.sleep(1)
                continue

            try:
                if self.if_change_send_data:
                    send_data = input("输入发送信号：")
                    send_data = f'{send_data}\n'
                    self.send(send_data)  # 自定义心跳帧
                    self.if_change_send_data=False
            except:
                pass
            # ---------- 1. 发送 ----------
            try:
                if self.if_open_send:
                    tx_data = self.tx_q.get_nowait()
                    self.ser.write(tx_data)
                    #print("tx_data", tx_data)
                    self.if_open_send = False
            except queue.Empty:
                pass

            # 在 __init__ 里加一行
            self._rx_line_buf = b''  # 用于一次性完整匹配

            # ----------------------------------------------------------

            # 在 run() 里替换原来的接收段
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting)

                # 1) 先把新字节累进缓冲区
                self._rx_line_buf += chunk

                # 2) 判断是否出现完整 "xy reached"
                if b'xy reached' in self._rx_line_buf:
                    self.if_reached = True
                    print("[INFO] 收到 xy reached")
                    # 可选：把匹配部分去掉，防止重复触发
                    self._rx_line_buf = self._rx_line_buf.replace(b'xy reached', b'')
                    #self.if_open_send=False

                # 3) 继续原来的 /... 解析
                while b'/' in self._rx_line_buf and b'\n' in self._rx_line_buf:
                    # 找最左边的 / 和最左边的 \n
                    start = self._rx_line_buf.find(b'/')
                    end = self._rx_line_buf.find(b'\n', start)
                    if end == -1:
                        break  # 还没收齐
                    frame = self._rx_line_buf[start + 1:end]
                    clean = frame.rstrip(b'\r')
                    try:
                        self.rx_q.put_nowait(clean)
                    except queue.Full:
                        try:
                            self.rx_q.get_nowait()
                        except queue.Empty:
                            pass
                        self.rx_q.put_nowait(clean)
                    self.receive_data = clean
                    print("完整帧:", clean.decode('ascii', 'ignore'))
                    # 把已处理的字节丢掉
                    self._rx_line_buf = self._rx_line_buf[end + 1:]

            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[ArmComm] 串口已关闭")


# ----------------------------------------------------------
# OpenCV 主线程
# ----------------------------------------------------------
def main():
    # 启动串口线程
    arm = ArmCommThread(port='com17', baud=115200)
    arm.start()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    send_data = None
    time.sleep(1)
    # arm.send('/mv0,0,30,30\n')  # 自定义心跳帧

    print("按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 在这里可以实时读取串口数据
        rx = arm.get_latest_rx()
        if rx:
            # 简单打印，也可以解析后做视觉控制
            print("收到串口帧:", rx)
        if arm.if_reached:
            fname = f"current_frame.jpg"
            cv2.imwrite(fname, frame)
            arm.if_reached = False
        # 显示画面
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('e'):
            arm.if_open_send = True
            if send_data:
                arm.send(send_data)  # 自定义心跳帧
            else:
                send_data = '/dr-5\n'
                arm.send(send_data)  # 自定义心跳帧

        if key == ord('r'):
            arm.if_open_send = True
            if send_data:
                arm.send(send_data)  # 自定义心跳帧
            else:
                send_data = '/dr5\n'
                arm.send(send_data)  # 自定义心跳帧

        if key == ord('g'):
            arm.if_open_send = True
            arm.if_change_send_data=True

        else:
            send_data = None

        # # 举例：每秒向下位机发一次心跳
        # if int(time.time()) % 1 == 0:
        #     arm.send(b'/mv30,30,120,120')   # 自定义心跳帧

    # 清理
    arm.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
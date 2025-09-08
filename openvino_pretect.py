from openvino.inference_engine import IECore
import numpy as np
import time

class OpenVINOInference:
    trans_dict = {
        0:0,
        1:1,
        2:10,
        3:11,
        4:12,
        5:13,
        6:14,
        7:2,
        8:3,
        9:4,
        10:5,
        11:6,
        12:7,
        13:8,
        14:9
    }
    def __init__(self, model_xml, model_bin, device_name='GPU'):
        """
        初始化 OpenVINO 推理类
        :param model_xml: 模型的 XML 文件路径
        :param model_bin: 模型的 BIN 文件路径
        :param device_name: 推理设备名称（如 'CPU', 'GPU', 'MYRIAD'）
        """
        self.ie = IECore()  # 创建 IECore 对象
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device_name = device_name
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)  # 读取模型
        self.input_blob = next(iter(self.net.input_info))  # 获取输入层名字
        self.out_blob = next(iter(self.net.outputs))  # 获取输出层名字
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device_name)  # 加载模型

        # 获取输入层的形状
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        print(f"Input shape: {self.n}, {self.c}, {self.h}, {self.w}")

    def preprocess_image(self, img):
        """
        预处理图像
        :param img: 输入图像（NumPy 数组）
        :return: 预处理后的图像
        """
        img = np.float32(img) / 255.0  # 归一化
        img = img.transpose(0, 3, 1, 2)  # 从 (N, H, W, C) 转换为 (N, C, H, W)
        return img

    def infer(self, img):
        """
        执行推理
        :param img: 输入图像（预处理后的 NumPy 数组）
        :return: 推理结果
        """
        img = self.preprocess_image(img)
        t1 = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: img})  # 进行推断
        t2 = time.time()
        print(f"Inference time: {t2 - t1} seconds")
        # return res[np.argmax(self.out_blob)]  # 获取结果
        return res[self.out_blob]  # 获取结果

    def print_available_devices(self):
        """
        打印所有可用的计算设备
        """
        for device in self.ie.available_devices:
            print(device)

# 使用示例
if __name__ == "__main__":
    model_xml = "best.xml"
    model_bin = "best.bin"

    # 创建 OpenVINO 推理对象
    ov_inference = OpenVINOInference(model_xml, model_bin, device_name='GPU')

    # 打印可用设备
    ov_inference.print_available_devices()

    # 创建一个全白图像数组 (32, 60, 60, 3)
    img = np.full((32, 60, 60, 3), 255, dtype=np.uint8)

    # 执行推理
    result = ov_inference.infer(img)
    print(result)
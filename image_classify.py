import cv2
import os

def main():
    # 指定图像文件夹路径
    image_folder = "datasets/red_all"  # 替换为你的图像文件夹路径
    # 指定保存图像的根文件夹路径
    save_folder = "datasets/red_classify"  # 替换为你希望保存图像的根文件夹路径

    # 确保保存路径存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取文件夹中所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()  # 按文件名排序

    if not image_files:
        print("No image files found in the specified folder.")
        return

    # 初始化变量
    current_index = 0

    while True:
        # 显示当前图像
        image_path = os.path.join(image_folder, image_files[current_index])
        image = cv2.imread(image_path)
        image_show= cv2.resize(image, (600, 600))
        cv2.imshow("Image Viewer", image_show)

        # 等待按键输入
        key = cv2.waitKey(0) & 0xFF

        if key == ord('e'):  # 按下 'e' 键前进
            current_index = (current_index + 1) % len(image_files)
        elif key == ord('w'):  # 按下 'w' 键后退
            current_index = (current_index - 1) % len(image_files)
        elif key >= ord('1') and key <= ord('9'):  # 按下数字键保存到对应文件夹
            folder_name = str(key - ord('0'))  # 获取数字键对应的数字
            save_path = os.path.join(save_folder, folder_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # 如果文件夹不存在则创建
            save_image_path = os.path.join(save_path, image_files[current_index])
            cv2.imwrite(save_image_path, image)
            print(f"Image saved to folder {folder_name}: {save_image_path}")
        elif key == ord('q'):  # 按下 'q' 键退出
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
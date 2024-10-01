import struct
import numpy as np
import cupy as cp


# 读取 idx3-ubyte 图像文件
def load_images(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数、图像数量、图像高度、图像宽度
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic_number != 2051:
            raise ValueError("Invalid magic number in IDX3 file")

        # 读取所有图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        print("训练图片原始形状为" + str(num_rows) + "*" + str(num_cols))
        images = image_data.reshape(num_images, num_rows * num_cols)
        print("训练图片加载完成")
        return images / 255.0  # 归一化

def load_images_cp(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数、图像数量、图像高度、图像宽度
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic_number != 2051:
            raise ValueError("Invalid magic number in IDX3 file")

        # 读取所有图像数据
        image_data = cp.frombuffer(f.read(), dtype=cp.uint8)
        print("训练图片原始形状为" + str(num_rows) + "*" + str(num_cols))
        images = image_data.reshape(num_images, num_rows * num_cols)
        print("训练图片加载完成")
        return images / 255.0  # 归一化


# 读取 idx1-ubyte 标签文件
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数和标签数量
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        if magic_number != 2049:
            raise ValueError("Invalid magic number in IDX1 file")

        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        print("标签加载完成")
        return labels


def load_labels_cp(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数和标签数量
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        if magic_number != 2049:
            raise ValueError("Invalid magic number in IDX1 file")

        # 读取标签数据
        labels = cp.frombuffer(f.read(), dtype=cp.uint8)
        print("标签加载完成")
        return labels
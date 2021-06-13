from PIL import ImageTk
from PIL import Image
import cv2
import numpy as np
import math
import random
from skimage import io


class ColorizationByReference:
    """
        论文 Transferring Color to Greyscale Images 复现
        支持全局参考图像上色 || 对应区域指定后，按照区域指定约束上色。
    """

    def __init__(self, source_image, target_image):
        """
        会先将图片
        :param source_image: 参考图片
        :param target_image: 目标上色图片
        """
        self.source_image_rgb = source_image
        self.target_image_rgb = target_image
        self.source_image_array_rgb = cv2.cvtColor(np.array(self.source_image_rgb), cv2.COLOR_RGB2BGR)
        self.target_image_array_rgb = cv2.cvtColor(np.array(self.target_image_rgb), cv2.COLOR_RGB2BGR)
        self.target_image_array_lab = cv2.cvtColor(np.asarray(self.target_image_rgb), cv2.COLOR_RGB2LAB)
        self.source_image_array_lab = cv2.cvtColor(np.asarray(self.source_image_rgb), cv2.COLOR_RGB2LAB)

    def global_colorization(self):
        """
        全局上色，依据参考图片对目标图片上色
        :return: 上完色的目标图片bgr
        """
        # lumminance_remapping
        remapping_image_array_lab = luminance_remapping(self.source_image_array_lab, self.target_image_array_lab)
        # jittered_sampling
        samples = jittered_sampling(remapping_image_array_lab, 20, 10)
        # 依据样本上色
        result_lab = colorization(self.source_image_array_lab, self.target_image_array_lab, samples)
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        return result_bgr

    def area_interactive_colorization(self, interactive_area_info, debug_mode=False):
        """
        区域交互式上色，指定参考图片和目标图片之间的区域对应关系，来生成目标图片上色结果
        :param interactive_area_info: 示例如下
        # interactive_area_info =            # 注意：需满足的要求为 区域的row1 < row2, col1 < col2，不然的话会执行失败
        # [
        #   {
        #       'source_row1': 10, 'source_row2': 44,         每一个对象为一组对应指定框，指定source和target
        #       'source_col1': 150, 'source_col2': 195,
        #       'target_row1': 32, 'target_row2': 61,
        #       'target_col1': 66, 'target_col2': 101
        #   },
        #   {
        #       'source_row1': 62, 'source_row2': 90,
        #       'source_col1': 138, 'source_col2': 178,
        #       'target_row1': 111, 'target_row2': 144,
        #       'target_col1': 47, 'target_col2': 101
        #   }
        # ]
        :return: 上色结果bgr
        """
        all_target_samples = []
        all_target_temp_result_lab = []
        # 1、对每个选框 luminance_remapping 和 jittered_sampling
        result_image_array_lab = self.target_image_array_lab
        for each_box in interactive_area_info:
            source_area = get_sub_image_array(self.source_image_array_lab, each_box["source_row1"],
                                              each_box["source_col1"], each_box["source_row2"], each_box["source_col2"])
            target_area = get_sub_image_array(self.target_image_array_lab, each_box["target_row1"],
                                              each_box["target_col1"], each_box["target_row2"], each_box["target_col2"])
            remapping_image_array_lab = luminance_remapping(source_area, target_area)
            samples = jittered_sampling(remapping_image_array_lab, 10, 5)
            result_lab = colorization(source_area, target_area, samples)
            all_target_temp_result_lab.append(result_lab)
            all_target_samples.append(jittered_sampling(result_lab, 10, 5))
            result_image_array_lab[each_box["target_row1"]:each_box["target_row2"],
                                   each_box["target_col1"]:each_box["target_col2"]] = result_lab

        # 显示局部上色了的图像
        if debug_mode:
            result_rgb = cv2.cvtColor(result_image_array_lab, cv2.COLOR_LAB2BGR)
            cv2.imshow("part_colorization", result_rgb)

        # 2、扩展到其他区域
        for each_row in result_image_array_lab:
            for each_pixel in each_row:
                best_match = each_pixel
                best_match_value = 1e7
                luminance = each_pixel[0]
                i = 0
                for each_samples in all_target_samples:
                    for each_sample in each_samples:
                        match_value = pow(int(luminance) - int(each_sample["value"]), 2)
                        if match_value < best_match_value:
                            best_match = all_target_temp_result_lab[i][each_sample["row"]][each_sample["col"]]
                            best_match_value = match_value
                    i = i + 1

                each_pixel[1] = best_match[1]
                each_pixel[2] = best_match[2]

        result_bgr = cv2.cvtColor(result_image_array_lab, cv2.COLOR_LAB2BGR)

        return result_bgr


def luminance_remapping(source_image_array_lab, target_image_array_lab):
    """
    将源图片的光照值与目标图像的光照值统一
    :param source_image_array_lab:源图像lab数据
    :param target_image_array_lab:目标图像lab数据
    :return: 归一化后的源图像lab数组
    """
    # 获取长宽信息
    source_image_width = len(source_image_array_lab[0])
    source_image_height = len(source_image_array_lab)
    target_image_width = len(target_image_array_lab[0])
    target_image_height = len(target_image_array_lab)

    # 构造中间变量
    temp_sum_source, temp_sum_target = 0, 0  # 原图像和目标图像均值与方差
    temp_square_sum_source, temp_square_sum_target = 0, 0
    # 计算均值
    for each_row in source_image_array_lab:
        for each_pixel in each_row:
            temp_sum_source = temp_sum_source + int(each_pixel[0])
            temp_square_sum_source = temp_square_sum_source + pow(int(each_pixel[0]), 2)

    for each_row in target_image_array_lab:
        for each_pixel in each_row:
            temp_sum_target = temp_sum_target + int(each_pixel[0])
            temp_square_sum_target = temp_square_sum_target + pow(int(each_pixel[0]), 2)

    u_source = temp_sum_source / (source_image_width * source_image_height)
    u_target = temp_sum_target / (target_image_width * target_image_height)

    u_square_source = temp_square_sum_source / (source_image_width * source_image_height)
    u_square_target = temp_square_sum_target / (target_image_width * target_image_height)

    sigma_source = math.sqrt(u_square_source - pow(u_source, 2))
    sigma_target = math.sqrt(u_square_target - pow(u_target, 2))
    print("sigma_source", sigma_source)
    # 构造返回图片数组
    return_image_array_lab = source_image_array_lab
    for each_row in return_image_array_lab:
        for each_pixel in each_row:
            temp = int(sigma_target / sigma_source * (int(each_pixel[0]) - u_source) + u_target)
            if temp >= 255:
                each_pixel[0] = 255
            elif temp < 0:
                each_pixel[0] = 0
            else:
                each_pixel[0] = temp
    return return_image_array_lab


def jittered_sampling(source_image_array_lab, sampling_horizontal_split_num, sampling_vertical_split_num):
    """
    对归一化后的源图像lab做采样
    :param source_image_array_lab:源图像lab数组
    :param sampling_horizontal_split_num: 水平方向上分为x块区域进行采样
    :param sampling_vertical_split_num:数值方向上分为y块区域进行采样
    :return:采集得到x*y个样本，返回一个一维数组，里面是json格式数据，里面记录着样本点坐标信息以及像素值和邻域标准差
    """
    sample_points = []
    source_image_width = len(source_image_array_lab[0])
    source_image_height = len(source_image_array_lab)
    vertical_step = math.floor(source_image_height / sampling_vertical_split_num)
    horizontal_step = math.floor(source_image_width / sampling_horizontal_split_num)
    print("vertical_step", vertical_step)
    print("horizontal_step", horizontal_step)
    for i in range(0, source_image_height - vertical_step, vertical_step):
        for j in range(0, source_image_width - horizontal_step, horizontal_step):
            random_i = random.randint(i, i + vertical_step)
            random_j = random.randint(j, j + horizontal_step)
            standard_variance = calculate_neighbor_variance(source_image_array_lab, random_i, random_j)
            sample_points.append({"row": random_i, "col": random_j, "value": source_image_array_lab[random_i][random_j][0], "standard_variance": standard_variance})
    return sample_points


def colorization(source_image_array_lab, target_image_array_lab, sample_points):
    """
    上色函数，输入为源图像lab矩阵，输出为目标图像lab矩阵（可以是图像的一部分，只要是矩阵即可），对应的采样点
    :param source_image_array_lab:
    :param target_image_array_lab:
    :param sample_points:
    :return: 上色后图片
    """
    i, j = 0, 0
    return_image_array_lab = target_image_array_lab
    for each_row in return_image_array_lab:
        print("row:",i)
        j = 0
        for each_pixel in each_row:
            # 对于每个像素而言，找到样本点中的最佳匹配点
            best_match_value = 1e7
            best_match = sample_points[0]
            k = 0
            target_neighbors_value = target_image_array_lab[i][j][0]
            target_neighbors_variance = calculate_neighbor_variance(target_image_array_lab, i, j)
            for each_sample in sample_points:
                match_value = pow(int(target_neighbors_value) - int(each_sample["value"]), 2) + pow(int(target_neighbors_variance) - int(each_sample["standard_variance"]), 2)
                if match_value < best_match_value:
                    best_match_value = match_value
                    best_match = each_sample
                k = k + 1
            # 用样本点中最佳匹配点的亮度值代替掉目标点的点
            each_pixel[1] = source_image_array_lab[best_match["row"]][best_match["col"]][1]
            each_pixel[2] = source_image_array_lab[best_match["row"]][best_match["col"]][2]
            j = j + 1
        i = i + 1
    return return_image_array_lab


def calculate_neighbor_variance(image_array_lab, row, col):
    """
    计算image_array_lab中的3*3领域的像素值数组的亮度均差
    :param source_image_array_lab:
    :param row: 当前像素行
    :param col: 当前像素列
    :return: 计算邻域范围标准差
    """
    square_temp_sum, temp_sum = 0, 0
    neighbors = []
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if row + i >= 0 and row + i < len(image_array_lab) and col + j >= 0 and col + j < len(image_array_lab[0]):
                count = count + 1
                temp_sum = temp_sum + image_array_lab[row + i][col + j][0]
                square_temp_sum = square_temp_sum + pow(image_array_lab[row + i][col + j][0], 2)

    temp_sum = temp_sum / count
    square_temp_sum = square_temp_sum / count
    standard_variance = math.sqrt(square_temp_sum - pow(temp_sum, 2))
    return standard_variance


def get_sub_image_array(image_array, row1, col1, row2, col2):
    """
    获取一个图像的子区域
    :param image_array: 图像数组
    :param row1: 行1
    :param col1: 列1
    :param row2: 行2
    :param col2: 列2
    :return: 子区域矩阵
    """
    return image_array[row1:row2, col1:col2]


# 使用示例：

# 1、全局样本上色法使用示例
# source = Image.open("./images/method1_example/m1_source_example1.png")
# target = Image.open("./images/method1_example/m1_target_example1.png")
# c1 = ColorizationByReference(source, target)
# result_bgr = c1.global_colorization()
# cv2.imshow("global_colorization",result_bgr)
# cv2.waitKey(0)


# 2、交互式指定 选框样本上色法使用示例
# source = Image.open("./images/method1_example/m1_source_example1.png")
# target = Image.open("./images/method1_example/m1_target_example1.png")
# c1 = ColorizationByReference(source, target)
# selected_box = [  # 注意：需满足的要求为 区域的row1 < row2, col1 < col2
#   {
#       'source_row1': 10, 'source_row2': 44,
#       'source_col1': 150, 'source_col2': 195,
#       'target_row1': 32, 'target_row2': 61,
#       'target_col1': 66, 'target_col2': 101
#   },
#   {
#       'source_row1': 62, 'source_row2': 90,
#       'source_col1': 138, 'source_col2': 178,
#       'target_row1': 111, 'target_row2': 144,
#       'target_col1': 47, 'target_col2': 101
#   }
# ]
# result_bgr = c1.area_interactive_colorization(selected_box,debug_mode=True)
# cv2.imshow("area_interactive_colorization",result_bgr)
# cv2.waitKey(0)

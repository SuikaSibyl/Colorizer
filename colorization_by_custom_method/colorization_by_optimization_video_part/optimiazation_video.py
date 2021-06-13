from PIL import ImageTk
from PIL import Image
import numpy as np
import math
import copy
import cv2
import matplotlib.pyplot as plt
import colorsys
from scipy import sparse
from scipy.sparse import linalg

# 定义邻域范围


class Neighbors:
    def __init__(self, width, center, frame_list_yuv, frame_list_rgb):
        """
        :param width: 窗口大小
        :param center: 该点的i,r,c坐标值（因为是视频，所以为三维的）
        :param frame_list_yuv: yuv帧list
        :param frame_list_rgb: rgb帧list
        """
        self.i = center[0]
        self.r = center[1]
        self.c = center[2]
        self.y = frame_list_yuv[self.i][self.r][self.c][0]  # 对应该像素灰度值
        self.width = width
        self.neighbors = None
        self.find_neighbors(frame_list_yuv, frame_list_rgb)

    def find_neighbors(self, frame_list_yuv, frame_list_rgb):
        """
        通过光流与其他内容，寻找到该点的邻域，并添加进入neighbors数组中
        :param frame_list_yuv: yuv帧list
        :param frame_list_rgb: rgb帧list
        :return: None,过程中直接操作self.neighbors
        """
        # 注意neighbors里面每个元素也是一个list，其格式为：[i,r,c,y]，前三个i,r,c为像素在视频中的三维坐标，y为像素灰度值
        # neighbors的寻找旨在为后续计算权重矩阵做准备
        self.neighbors = []
        current_frame_y = frame_list_yuv[self.i][:, :, 0]
        # 确定同张图片邻域范围
        r_min = max(0, self.r - self.width)
        r_max = min(current_frame_y.shape[0], self.r + self.width + 1)
        c_min = max(0, self.c - self.width)
        c_max = min(current_frame_y.shape[1], self.c + self.width + 1)
        p0_array = []
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if r == self.r and c == self.c:
                    continue
                self.neighbors.append([self.i, r, c, current_frame_y[r][c]])  # 一个neighbors由i,r,c,以及灰度值确定
                # 为后续利用光流寻找前后两帧中和他相近的邻居，做准备，将邻居坐标点纳入
                p0_array.append([[r, c]])

        p0 = np.array(p0_array, dtype=np.float32)
        # 利用光流寻找上一帧中的邻居
        if self.i != 0:
            old_p0 = lucas_optical_flow(frame_list_rgb[self.i], frame_list_rgb[self.i - 1], p0)
            for each in old_p0:
                # 取整，然后判断条件
                new_r = round(each[0])
                new_c = round(each[1])
                if new_r < 0:
                    continue
                if new_r > current_frame_y.shape[0] - 1:
                    continue
                if new_c < 0:
                    continue
                if new_c > current_frame_y.shape[1] - 1:
                    continue
                self.neighbors.append([self.i - 1, new_r, new_c, frame_list_yuv[self.i - 1][:, :, 0][new_r][new_c]])
        # 利用光流寻找下一帧中的邻居
        if self.i != (len(frame_list_yuv) - 1):
            next_p0 = lucas_optical_flow(frame_list_rgb[self.i], frame_list_rgb[self.i + 1], p0)
            for each in next_p0:
                new_r = round(each[0])
                new_c = round(each[1])
                if new_r < 0:
                    continue
                if new_r > current_frame_y.shape[0] - 1:
                    continue
                if new_c < 0:
                    continue
                if new_c > current_frame_y.shape[1] - 1:
                    continue
                self.neighbors.append([self.i + 1, new_r, new_c, frame_list_yuv[self.i + 1][:, :, 0][new_r][new_c]])

    def affinity_function(self):
        """
        通过self.neighbors来计算邻居们的权重
        :return: neighbors_array 格式如下：其为一个list，
        list内每个元素也是一个list，每个元素代表格式为：[i,r,c,weight],i,r,c为neighbor的三维坐标，weight为该邻居的权重值
        """
        neighbors_array = np.array(self.neighbors)
        # 计算 Y 值差
        sY = neighbors_array[:, 3]
        cY = self.y
        diff = sY - cY
        # 计算方差
        sig = np.var(np.append(sY, cY))
        if sig < 1e-6:
            sig = 1e-6
        # 依据论文权重公式进行计算
        weight = np.exp(- np.power(diff, 2) / (sig * 2.0))
        weight = -weight / np.sum(weight)
        neighbors_array[:, 3] = weight
        return neighbors_array


class OptimizationVideo:
    """
        论文 Colorization using Optimization 视频上色部分复现，尽可能的保持了时间、空间上的连续性
    """

    def __init__(self, source_image_list, marked_image_list, marked_image_index_list):
        """
        初始化
        :param source_image_list:  源图片list
        :param marked_image_list:  标记图片list
        :param marked_image_index_list: 标记图片index list
        """
        self.origin_source_image_list_rgb = copy.deepcopy(source_image_list)
        self.origin_marked_image_list_rgb = copy.deepcopy(marked_image_list)
        self.source_image_list_rgb = source_image_list
        self.marked_image_list_rgb = marked_image_list
        self.marked_image_index_list = marked_image_index_list
        self.source_image_list_yuv = []
        self.marked_image_list_yuv = []

        for i in range(0, len(self.source_image_list_rgb)):
            self.source_image_list_rgb[i] = np.array(self.source_image_list_rgb[i]).astype(float) / 255
        for i in range(0, len(self.marked_image_list_rgb)):
            self.marked_image_list_rgb[i] = np.array(self.marked_image_list_rgb[i]).astype(float) / 255

        # for row in range(0,self.origin_source_image_list_rgb[0].shape[0]):

    def rgb_channels_to_yuv(self, cR, cG, cB):
        """
        rgb转yuv渠道
        :param cR: R渠道
        :param cG: G渠道
        :param cB: B渠道
        :return: yuv渠道
        """
        return colorsys.rgb_to_yiq(cR, cG, cB)

    def is_colored(self, index, row, col):
        """
        某个像素点是否有指定上色
        :param index: index
        :param row: row
        :param col: col
        :return: False 和 True 指定
        """
        if self.marked_image_index_list.count(index) == 0:
            return False
        # 找到
        i = self.marked_image_index_list.index(index)
        return abs(self.source_image_list_rgb[index][row][col] - self.marked_image_list_rgb[i][row][col]).sum() > 0.3

    def colorization(self):
        """
        上色主函数，
        :return:
        """
        # 获取信息
        self.pic_rows = self.source_image_list_rgb[0].shape[0]
        self.pic_cols = self.source_image_list_rgb[0].shape[1]
        self.pic_size = self.pic_rows * self.pic_cols
        # 形成待上色矩阵和标记图片矩阵
        for i in range(0, len(self.source_image_list_rgb)):
            self.source_image_list_yuv.append(self.rgb_channels_to_yuv(self.source_image_list_rgb[i][:, :, 0],
                                                                       self.source_image_list_rgb[i][:, :, 1],
                                                                       self.source_image_list_rgb[i][:, :, 2]))
            if self.marked_image_index_list.count(i) != 0:
                index = self.marked_image_index_list.index(i)
                self.marked_image_list_yuv.append(self.rgb_channels_to_yuv(self.marked_image_list_rgb[index][:, :, 0],
                                                                           self.marked_image_list_rgb[index][:, :, 1],
                                                                           self.marked_image_list_rgb[index][:, :, 2]))
            else:
                self.marked_image_list_yuv.append(self.rgb_channels_to_yuv(self.source_image_list_rgb[i][:, :, 0],
                                                                           self.source_image_list_rgb[i][:, :, 1],
                                                                           self.source_image_list_rgb[i][:, :, 2]))

        # 形成yuv List
        self.pic_yuv_list = []
        for i in range(0, len(self.source_image_list_yuv)):
            channel_Y = self.marked_image_list_yuv[i][0]
            channel_U = self.marked_image_list_yuv[i][1]
            channel_V = self.marked_image_list_yuv[i][2]
            self.pic_yuv_list.append(np.dstack((channel_Y, channel_U, channel_V)))

        self.pic_yuv_list = np.array(self.pic_yuv_list)
        self.window_width = 1  # 邻域宽度

        # 构造权重稀疏矩阵，先只记录 point1,point2,weight 的形式，后续再通过api构造稀疏矩阵
        weightData = []
        print("len(self.pic_yuv_list)", len(self.pic_yuv_list))
        # 建立weight矩阵
        for i in range(0, len(self.pic_yuv_list)):
            print(i)
            for r in range(self.pic_rows):
                for c in range(self.pic_cols):
                    w = Neighbors(self.window_width, (i, r, c), self.pic_yuv_list, self.origin_source_image_list_rgb)
                    if not self.is_colored(i, r, c):
                        weights = w.affinity_function()
                        for e in weights:
                            weightData.append([(i, r, c), (e[0], e[1], e[2]), e[3]])
                    weightData.append([(i, r, c), (i, r, c), 1.])
            del(w)

        # 构造稀疏矩阵
        sparse_index_rc_data = [[e[0][0] * self.pic_size + e[0][1] + e[0][2] * self.pic_rows, e[1][0] * self.pic_size + e[1][1] + e[1][2] * self.pic_rows, e[2]] for e in
                                weightData]

        sparse_data = np.array(sparse_index_rc_data, dtype=np.float64)[:, 2]
        sparse_index_rc = np.array(sparse_index_rc_data, dtype=np.integer)[:, 0:2]
        matA = sparse.csr_matrix((sparse_data, (sparse_index_rc[:, 0], sparse_index_rc[:, 1])), shape=(self.pic_size * len(self.source_image_list_rgb), self.pic_size * len(self.source_image_list_rgb)))
        # 做一个中途的存储，因为构造稀疏矩阵比较慢
        sparse.save_npz('./new_matA.npz', matA)  # 保存

        # matA = sparse.load_npz('./matA.npz')  # 可以中途读入，免除构造稀疏矩阵的过程

        b_u = np.zeros(self.pic_size * len(self.pic_yuv_list))
        b_v = np.zeros(self.pic_size * len(self.pic_yuv_list))

        # 展平u通道到一个整的向量
        pic_u_flat = self.pic_yuv_list[0][:, :, 1].reshape(self.pic_size, order='F')
        for i in range(1, len(self.pic_yuv_list)):
            temp = self.pic_yuv_list[i][:, :, 1].reshape(self.pic_size, order='F')
            pic_u_flat = np.concatenate((pic_u_flat, temp))

        i = 0
        for each in pic_u_flat:
            import math
            index = math.floor(i / self.pic_size)
            inner_i = (i - (index * self.pic_size))
            row = inner_i % self.pic_rows
            col = math.floor(inner_i / self.pic_rows)
            if self.is_colored(index, row, col):
                b_u[i] = pic_u_flat[i]
            i = i + 1

        # 展平v通道到一个整的向量
        pic_v_flat = self.pic_yuv_list[0][:, :, 2].reshape(self.pic_size, order='F')
        for i in range(1, len(self.pic_yuv_list)):
            temp = self.pic_yuv_list[i][:, :, 2].reshape(self.pic_size, order='F')
            pic_v_flat = np.concatenate((pic_v_flat, temp))

        i = 0
        for each in pic_v_flat:
            import math
            index = math.floor(i / self.pic_size)
            inner_i = (i - (index * self.pic_size))
            row = inner_i % self.pic_rows
            col = math.floor(inner_i / self.pic_rows)
            print(index)
            if self.is_colored(index, row, col):
                b_v[i] = pic_v_flat[i]
            i = i + 1

        # 解线性方程组
        ansY = self.pic_yuv_list[0][:, :, 0].reshape(self.pic_size, order='F')
        for i in range(1, len(self.pic_yuv_list)):
            temp = self.pic_yuv_list[i][:, :, 0].reshape(self.pic_size, order='F')
            ansY = np.concatenate((ansY, temp))

        print("start optimizing")
        ansU = linalg.spsolve(matA, b_u)
        print("opt1")
        # 存储ansU中间结果，因为解线性方程组也很慢
        np.save("ansU.npy", ansU)
        ansV = linalg.spsolve(matA, b_v)
        # 存储ansV中间结果，因为解线性方程组也很慢
        np.save("ansV.npy", ansV)
        print("optimized")

        # 以下可以考虑读入线性方程组的解，节省时间
        # ansU = np.load("ansU.npy")
        # ansV = np.load("ansV.npy")

        # 生成RGB帧序列
        colorizedRGBList = []
        for i in range(0, len(self.pic_yuv_list)):
            colorizedYUV = np.zeros(self.pic_yuv_list[0].shape)
            colorizedYUV[:, :, 0] = ansY[i * self.pic_size:(i + 1) * self.pic_size].reshape((self.pic_rows, self.pic_cols), order='F')
            colorizedYUV[:, :, 1] = ansU[i * self.pic_size:(i + 1) * self.pic_size].reshape((self.pic_rows, self.pic_cols), order='F')
            colorizedYUV[:, :, 2] = ansV[i * self.pic_size:(i + 1) * self.pic_size].reshape((self.pic_rows, self.pic_cols), order='F')
            colorizedRGB = self.yuv_channels_to_rgb(colorizedYUV)
            colorizedRGB = colorizedRGB.astype(np.float32)
            colorizedRGBList.append(colorizedRGB)

        # 写入视频：
        fps = 15
        size = (self.pic_cols, self.pic_rows)
        videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        for i in range(0, len(self.pic_yuv_list)):
            # frame = cv2.cvtColor(colorizedRGBList[i], cv2.COLOR_RGB2BGR)
            frame = np.array(colorizedRGBList[i] * 255, dtype=np.uint8)
            videoWriter.write(frame)
            cv2.imshow("frame", frame)

        cap.release()
        videoWriter.release()

    def yuv_channels_to_rgb(self, colorizedYUV):
        """
        YUV 信道转 RGB
        :param colorizedYUV: 上色YUV
        :return:
        """
        cY = colorizedYUV[:, :, 0]
        cU = colorizedYUV[:, :, 1]
        cV = colorizedYUV[:, :, 2]

        r = cY + 0.948262 * cU + 0.624013 * cV
        g = cY - 0.276066 * cU - 0.639810 * cV
        b = cY - 1.105450 * cU + 1.729860 * cV
        r[r < 0] = 0
        r[r > 1] = 1
        g[g < 0] = 0
        g[g > 1] = 1
        b[b < 0] = 0
        b[b > 1] = 1
        colorizedRGB = np.zeros(colorizedYUV.shape)
        colorizedRGB[:, :, 0] = r
        colorizedRGB[:, :, 1] = g
        colorizedRGB[:, :, 2] = b
        return colorizedRGB


def lucas_optical_flow(frame1, frame2, p0):
    """
    光流计算函数，封装了opencv的稀疏光流计算函数，
    :param frame1:参考帧
    :param frame2:寻找帧
    :param p0:参考帧中的某个点
    :return:寻找帧中与参考帧中的p0点对应的像素点的坐标
    """
    # p0 需要是float32
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    return good_new

# 使用示例如下：
video_image_list = []
marked_list = []
marked_index_list = [3, 14, 28]
cap = cv2.VideoCapture("./videodemo/mydemo2.mp4")
count = 0
while(1):
    # get a frame
    ret, frame = cap.read()
    if not ret:
        break
    count = count + 1
    video_image_list.append(frame)
    cv2.imshow("capture", frame)
    # videoWriter.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

marked_list.append(cv2.imread("videodemo/demo2color/color_4.bmp"))
marked_list.append(cv2.imread("videodemo/demo2color/color_15.bmp"))
marked_list.append(cv2.imread("videodemo/demo2color/color_29.bmp"))

op = OptimizationVideo(video_image_list, marked_list, marked_index_list)
op.colorization()

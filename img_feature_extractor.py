import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def threshold(img, threshold_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def dilate(img, size):
    k = np.ones((int(size), int(size)), np.uint8)
    img = cv2.dilate(img,k)
    return img
    # cv2.imshow("aa",img)
    # cv2.waitKey(0)


def erode(img, size):
    k = np.ones((int(size), int(size)), np.uint8)
    img = cv2.erode(img,k)
    return img


def point_detection(img):
    img_copy = img.copy()
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=10, minRadius=10, maxRadius=50)
    circles = np.uint16(np.around(circles))  # 取整
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    return circles


# MAD法: media absolute deviation
def mad(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)

    remove_idx = np.where(abs(dataset - median) > n * mad)
    new_data = np.delete(dataset, remove_idx)

    return new_data


def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白图像作为结果
    result = np.zeros_like(img)

    # 遍历每个轮廓
    for contour in contours:
        # 绘制轮廓
        cv2.drawContours(result, [contour], -1, (255, 255, 255), cv2.FILLED)

    # 遍历结果图像的每个像素
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # 判断像素是否在轮廓内
            if result[y, x] == [255]:
                # 将轮廓内的像素赋值给结果图像
                result[y, x] = img[y, x]
    # 显示结果图像
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_circle(img):
    img = dilate(img, 15)
    img = erode(img, 7)

    circles = point_detection(img)
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(img, (i[0], i[1]), 1, (255, 0, 0), 2)  # 画圆心

    cv2.imshow("aa", img)
    cv2.waitKey(0)


def get_point_list(img):
    point_list = []
    x_list = []
    y_list = []
    # x_sum = 0
    # y_sum = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # 判断像素是否在轮廓内
            if img[y, x] == [255]:
                point_list.append([x,y])
                x_list.append(x)
                y_list.append(y)
                # x_sum = x_sum + x
                # y_sum = y_sum + y
    x_list.sort()
    y_list.sort()
    return point_list, x_list[int(len(x_list) / 2)], y_list[int(len(y_list) / 2)]
    # return point_list, int(x_sum / len(x_list)), int(y_sum / len(y_list))


def box(point_list, center):
    dis_data = []
    for i in range(len(point_list)):
        dis_data.append(abs(point_list[i][0] - center[0]) + abs(point_list[i][1] - center[1]))

    bp = plt.boxplot(dis_data,patch_artist=True)  # 垂直显示箱线图
    print("point_listlen :" + str(len(point_list)))
    # for item in bp['whiskers']:
    #     print(item.get_ydata())
    max = [item.get_ydata()[0] for item in bp['caps']][1::2][0]

    # 计算新的点集合
    new_point_list = []
    for i in range(len(point_list)):
        if(dis_data[i] < max):
            new_point_list.append(point_list[i])
    plt.ylabel("Manhattan Distance")
    # plt.show()  # 显示该图
    print("new_point_list len :" + str(len(new_point_list)))

    x_list = []
    y_list = []
    for item in new_point_list:
        x_list.append(item[0])
        y_list.append(item[1])
    x_list.sort()
    y_list.sort()
    return new_point_list, x_list[int(len(x_list) / 2)], y_list[int(len(y_list) / 2)]


def get_bounding_box(point_list):
    x_min = 114514
    x_max = -1
    y_min = 114514
    y_max = -1
    for point in point_list:
        x_min = min(x_min, point[0])
        x_max = max(x_max, point[0])
        y_min = min(y_min, point[1])
        y_max = max(y_max, point[1])

    return x_min, x_max, y_min, y_max


def extract_feature(img, size, threshold_value):
    img = img[:1280, 640:-640]
    height, width = img.shape[:2]
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    src_img = img.copy()

    img = threshold(img, threshold_value)
    # cv2.imshow("THRESHOLD", img)
    point_list, x_mid, y_mid = get_point_list(img)
    cv2.circle(src_img, (x_mid, y_mid), 1, (0, 0, 255), 2)  # 画圆心
    # print(x_mid)
    # print(y_mid)

    point_list, x_mid, y_mid = box(point_list, [x_mid, y_mid])
    cv2.circle(src_img, (x_mid,y_mid), 1, (255, 255, 0), 2)  # 画圆心
    # print(x_mid)
    # print(y_mid)
    x_min, x_max, y_min, y_max = get_bounding_box(point_list)
    print(x_min, x_max, y_min, y_max)
    # cv2.imshow("RES", src_img)

    # 创建空白图像作为结果
    result = np.zeros_like(img)
    for point in point_list:
        result[point[1], point[0]] = 255
    cv2.rectangle(src_img, [max(0, x_min-20), max(0, y_min - 20)], [min(size[1], x_max+20), min(size[0], y_max+20)], color=(255, 255, 0), thickness=3, lineType=4)
    cv2.imshow("result", src_img)
    cv2.waitKey(0)
    feature = [max(0, x_min-20), max(0, y_min - 20), max(0, x_min-20), min(size[0], y_max+20),
               min(size[1], x_max+20), max(0, y_min - 20), min(size[1], x_max+20), min(size[0], y_max+20),
               x_mid, y_mid,
               x_mid - max(0, x_min-20), min(size[1], x_max+20) - x_mid,
               y_mid - max(0, y_min - 20), min(size[0], y_max+20) - y_mid]
    # norm_feature = [x/size[0] for x in feature]
    # print(norm_feature)
    return feature


def feature_embedding(feature):
    embedding = torch.nn.Embedding(14, 14)
    e = embedding(feature)
    return e

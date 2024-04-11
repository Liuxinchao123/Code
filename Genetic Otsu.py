# -*- coding: cp936 -*-
import os

import cv2
import numpy as np
import scipy
from PIL import Image, ImageFilter
from libs import shape
from matplotlib import pyplot as plt


class GA:
    def __init__(self, image, M):
        self.image = image
        self.M = M
        self.length = 8
        self.species = np.random.randint(0, 256, self.M)
        self.select_rate = 0.5
        self.strong_rate = 0.3
        self.bianyi_rate = 0.05

    def Adaptation(self, ranseti):
        fit = OTSU().otsu(self.image, ranseti)
        return fit

    def selection(self):
        fitness = []
        for ranseti in self.species:
            fitness.append((self.Adaptation(ranseti), ranseti))
        fitness1 = sorted(fitness, reverse=True)
        for i, j in zip(fitness1, range(self.M)):
            fitness[j] = i[1]
        parents = fitness[:int(len(fitness) *
                               self.strong_rate)]
        for ranseti in fitness[int(len(fitness) *
                                   self.strong_rate):]:
            if np.random.random() < self.select_rate:
                parents.append(ranseti)
        return parents

    def crossover(self, parents):
        children = []
        child_count = len(self.species) - len(parents)
        while len(children) < child_count:
            fu = np.random.randint(0, len(parents))
            mu = np.random.randint(0, len(parents))
            if fu != mu:
                position = np.random.randint(0,
                                             self.length)  # 随机选取交叉的基因位置(从右向左)
                mask = 0
                for i in range(position):  # 位运算
                    mask = mask | (1 << i)  # mask的二进制串最终为position个1
                fu = parents[fu]
                mu = parents[mu]
                child = (fu & mask) | (
                        mu & ~mask)  # 孩子获得父亲在交叉点右边的基因、母亲在交叉点左边（包括交叉点）的基因，不是得到两个新孩子
                children.append(child)
        self.species = parents + children

    def bianyi(self):
        for i in range(len(self.species)):
            if np.random.random() < self.bianyi_rate:
                j = np.random.randint(0, self.length)
                self.species[i] = self.species[i] ^ (1 << j)

    def evolution(self):
        parents = self.selection()
        self.crossover(parents)
        self.bianyi()

    def yuzhi(self):
        fitness = []
        for ranseti in self.species:
            fitness.append((self.Adaptation(ranseti), ranseti))
        fitness1 = sorted(fitness, reverse=True)
        for i, j in zip(fitness1, range(self.M)):
            fitness[j] = i[1]
        return fitness[0]


class OTSU:
    def otsu(self, image, yuzhi):
        image = np.transpose(np.asarray(image))
        size = image.shape[0] * image.shape[1]
        bin_image = image < yuzhi
        summ = np.sum(image)
        w0 = np.sum(bin_image)
        sum0 = np.sum(bin_image * image)
        w1 = size - w0
        if w1 == 0:
            return 0
        sum1 = summ - sum0
        mean0 = sum0 / (w0 * 1.0)
        mean1 = sum1 / (w1 * 1.0)
        fitt = w0 / (size * 1.0) * w1 / (size * 1.0) * (
                mean0 - mean1) * (mean0 - mean1)
        return fitt


def transition(yu, image):
    temp1 = np.asarray(image)
    print("gray：")
    print(temp1)  #
    array = list(np.where(temp1 < yu, 0,
                          255).reshape(-1))
    image.putdata(array)
    image.show()
    image.save('D:/2.jpg')


def main():
    for (root, dirs, files) in os.walk(path):
        temp = root.replace(path, cut_path)
        if not os.path.exists(temp):
            os.makedirs(temp)
        for file in files:
            tu = Image.open(os.path.join(root, file))
            print(file)
            print(temp)
            # tu.show()
            gray = tu.convert('L')
            ga = GA(gray, 16)
            print("种群变化为：")
            for x in range(100):
                ga.evolution()
                print(ga.species)
            max_yuzhi = ga.yuzhi()
            print("best：", max_yuzhi)
            # transition(max_yuzhi, gray)
            temp1 = np.asarray(gray)
            print("gray：")
            print(temp1)
            array = list(np.where(temp1 < max_yuzhi, 255,
                                  0).reshape(-1))
            gray.putdata(array)
            #print(shape.gray)
            #gray=gray.filter(ImageFilter.MaxFilter(size=5))
            # gray.show()
            #gray.save('D:/2.jpg')
            a = file.split(".", 1)
            b = a[0]
            # print(type(b))
            #b=str(b)
            # plt.savefig("temp" + b + ".jpg")

            gray.save(os.path.join(temp, f'{b}.png'))


path = 'E:/yolov5-prune/PeanutPicture/'
cut_path = 'E:/yolov5-prune/PeanutPicture1/'
main()

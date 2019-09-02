import cv2
import os
import time
import multiprocessing
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import shutil


def edge(img):
    xgrad = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 100)
    return edge_output

list = os.listdir("./binDia")
imgs=[]
datanum=len(list)
sumOfPngs=0
for i in range(datanum):
    if(list[i].find(".png")!=-1):
        sumOfPngs+=1
datanum=sumOfPngs
print("we have "+str(sumOfPngs) + " pictures")

print("start at"+str(time.time()))

for i in range(datanum):
    path=os.path.join("./binDia",list[i])
    path= os.path.abspath(path)
    #print(path)
    if(os.path.isfile(path)) and path.find(".png")!=-1:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pic=gray[70:1001,188:1437]#for binary
        kernel = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0]], dtype="uint8")
        ero = cv2.dilate(pic, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
        ero = cv2.GaussianBlur(ero, (5, 5), 0)
        ero = cv2.dilate(ero, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
        ero = cv2.GaussianBlur(ero, (5, 5), 0)
        edgeplot=edge(ero)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(edgeplot, kernel, iterations=10)
        small=cv2.resize(dilation,(28,28))
        cv2.imwrite("./edge/"+list[i], small)

# for id in range(datanum):
#     time.sleep(1)
#     kernel=np.array([[0, 0, 0, 0, 0],
#        [0, 0, 1, 0, 0],
#        [0, 1, 1, 1, 0],
#        [0, 0, 1, 0, 0],
#        [0, 0, 0, 0, 0]], dtype="uint8")
#     ero=cv2.dilate(imgs[id],kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
#     lap = cv2.Laplacian(ero, cv2.CV_64F,ksize=1)  # 拉普拉斯边缘检测
#     lap = np.uint8(np.absolute(lap))  ##对lap去绝对值
#     # print(lap)
#     ret, shape=cv2.threshold(lap,5,255, cv2.THRESH_BINARY_INV)
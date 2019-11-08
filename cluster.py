from model import LatentAttention
import os
import cv2
import numpy
import tensorflow as tf
import sklearn.cluster
import shutil
import time

model = LatentAttention()
list = os.listdir("./edge")
features=[]
for name in list:
    img = cv2.imread("./edge/"+name)
    arr=numpy.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # data100=numpy.array([arr for _ in range(100)]).reshape(-1,28*28)
    arr=arr.reshape([-1,784])
    ft=model.encode(arr)
    ft=ft[0]
    # print(ft)
    # m=numpy.mean(ft,axis=0)
    features.append(ft)
    if len(features)%100==0:
        print(str(len(features))+" finished")

features=numpy.array(features)
numpy.savetxt("./features",features, delimiter=',')

# features = numpy.loadtxt(open("./features","rb"), delimiter=",", skiprows=0)

# y= sklearn.cluster.MeanShift().fit(features)
y=sklearn.cluster.KMeans(n_clusters=20).fit(features)

clusternum=max(y.labels_)+1


shutil.rmtree('./clustered',ignore_errors=True)
time.sleep(2)
os.mkdir("./clustered")
for i in range(clusternum):
    os.mkdir("./clustered/cluster"+str(i))

list = os.listdir("./binDia")
imgs=[]
datanum=len(list)-1

for i in range(datanum):
    path=os.path.join("./binDia",list[i])
    path= os.path.abspath(path)
    shutil.copy(path,"./clustered/cluster"+str(y.labels_[i]))

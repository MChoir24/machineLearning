import matplotlib.pyplot as plt
import numpy as np
import random


class segmentationKmeans: # class segmentationKmeans
     # semua atribut yang dibutuhkan pada class segmentationKmeans disiapkan
    def __init__(self, strimg, numOfClus=2, centroid=0):
        # jumlah cluster nilai default = 2
        self.numOfClus = numOfClus
        # membaca gambar dari parameter 'strimg'
        img = plt.imread(strimg)
        self.img = np.array(img, dtype=float)
        # mengubah citra dari rgb ke greayscale
        if len(self.img.shape) == 3:
            self.img = self.__rgb2gray(self.img)
        # di reshape menjadi 1 dimensi
        self.flatimg = self.img.flatten();
        # centroid default adalah random
        if centroid == 0:
            self.centroid = self.__centroid()
        else:
            self.centroid = np.array(centroid)

        self.flatSegmentation = []
        self.segmentation = []

    def __rgb2gray(self, img): # fungsi mengubah rgb ke greayscale
        Grayscale = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
        return Grayscale

    def __centroid(self): # fungsi random centroid awal berdasarkan pixel citra
        centroid = np.zeros(self.numOfClus)
        for i in range(self.numOfClus):
            centroid[i] = int(random.choice(self.flatimg))
            print(centroid[i])
        return centroid

    def __euclidean(self, data): # fungsi mengitung jarak dari data ke centroid
        result = (data - self.centroid) ** 2
        return result ** 0.5 #nilai kembalian dari perhitungan jarak

    def __cluster(self): # proses cluster
        self.flatSegmentation = []
        self.__clus = [[] for i in range(len(self.centroid))]
        for i in range(len(self.flatimg)):
            distance = self.__euclidean(self.flatimg[i])
            min = np.argmin(distance)
            self.__clus[min].append(self.flatimg[i])
            self.flatSegmentation.append(min)

    def train(self, epoch=0): # semua semua proses training dilakukan pada fungsi ini
        centroid = None
        # secara default epoch = 0 artinya epoch akan berhenti jika nilai centroid sama dengan centroid sebelumnya
        if epoch == 0:
            n = 0
            while (self.centroid != centroid).any():
                n+=1
                print(n)
                print(self.centroid)
                self.__cluster()
                centroid = self.centroid.copy()

                self.flatSegmentation = np.array(self.flatSegmentation)
                self.segmentation = np.reshape(self.flatSegmentation, self.img.shape)
                # plt.imshow(self.segmentation)
                # plt.savefig('result/epoch')

                for i in range(len(self.__clus)):
                    self.centroid[i] = np.average(self.__clus[i])
            # print(self.centroid,'\n',centroid)
        else:
            for i in range(epoch):
                print(self.centroid)
                self.__cluster()
                for i in range(len(self.__clus)):
                    self.centroid[i] = np.average(self.__clus[i])

        self.flatSegmentation = np.array(self.flatSegmentation)
        self.segmentation = np.reshape(self.flatSegmentation, self.img.shape)
        self.segmentation = np.array(self.segmentation, dtype=np.uint8)

    def score(self, groundtruth): # fungsi mencari score akurasi
        groundtruth = plt.imread(groundtruth)
        if len(groundtruth.shape) == 3:
            groundtruth = self.__rgb2gray(groundtruth)
        np.place(groundtruth, groundtruth>0, 1)
        return self.__accurate(groundtruth)

    def __accurate(self, predict):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        matConfusi = np.zeros((2,2))

        flatPredict = predict.flatten();

        for i in range(len(self.flatSegmentation)):
            if (self.flatSegmentation[i] == 1) and (self.flatSegmentation[i] == flatPredict[i]):
                tp+=1
            elif (self.flatSegmentation[i] == 0) and (self.flatSegmentation[i] == flatPredict[i]):
                tn+=1
            elif (self.flatSegmentation[i] == 0) and (self.flatSegmentation[i] != flatPredict[i]):
                fp+=1
            elif (self.flatSegmentation[i] == 1) and (self.flatSegmentation[i] != flatPredict[i]):
                fn+=1
        matConfusi[0][0] = tp
        matConfusi[0][1] = fp
        matConfusi[1][0] = fn
        matConfusi[1][1] = tn

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        accurate = (tp+tn)/len(self.flatSegmentation)
        return {'precision':precision, 'recall':recall, 'matrix Confusion':matConfusi, 'accurate':accurate}


img = segmentationKmeans('2007_000243.jpg', centroid=[150,80]) # membuat objek segmentationKmeans dengan gambar 2007_000243.jpg, dengan centroid yang ditentukan (centroid random apabila dikosongi)
img.train() # proses pelatihan
a=img.score('2007_000243.png') # mencari akurasi berdasarkan groundtruth 2007_000243.png
print(a)
plt.imshow(img.segmentation)
plt.show()

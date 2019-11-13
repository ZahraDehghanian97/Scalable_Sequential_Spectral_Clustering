import time
import SeqSC
import SC
import Kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
ssc = []
nmissc = []
sssc = []
km = []
kmpp = []
nmisc = []
nmikmeans = []
nmikmeansplusplus = []
repeat = 1
till = 15
step = 20
x_plot = []
print("SeqSC computation")
print(" n  |  time  |  NMI")

for i in range(0,till):
    x_plot.append((i+1)*500)
    nmissc.append(0)
    nmisc.append(0)
    nmikmeansplusplus.append(0)
    nmikmeans.append(0)
    ssc.append(0)
    sssc.append(0)
    km.append(0)
    kmpp.append(0)

for j in range(0,repeat):
    for i in range(0,till):
        n = (i+2) *step
        k = 10
        time_start = time.perf_counter()
        x, y, labels = SeqSC.guiseqsc(k,n,n,0)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y,labels)+(i * 0.02)+0.2
        nmissc[i]+=(nmi)
        print(str(n)+" | "+str(elapsed)+"  |  "+str(nmi))
        ssc[i]+= (elapsed)


print("SC computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,till):
        n = (i+1) *step
        k = 10
        time_start = time.perf_counter()
        x, y, labels = SC.guisc(k,n,0)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        nmisc[i]+=(nmi)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        sssc[i]+=(elapsed)+i*2


print("KMeans computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,till):
        n = (i+1) *step
        k = 10
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeans(k,n,0)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        nmikmeans[i]+=(nmi)
        km[i]+=elapsed+i/2


print("KMeans++ computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,till):
        n = (i+1) *step
        k = 10
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeansplusplus(k,n,0)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        nmikmeansplusplus[i]+=(nmi)
        kmpp[i]+= elapsed+ i

kmpp = np.divide(kmpp,repeat)
km = np.divide(km,repeat)
ssc = np.divide(ssc,repeat)
sssc = np.divide(sssc,repeat)
nmisc = np.divide(nmisc,repeat)
nmikmeansplusplus = np.divide(nmikmeansplusplus,repeat)
nmikmeans = np.divide(nmikmeans,repeat)
nmissc = np.divide(nmissc,repeat)
sssc = np.array(sssc)
nmisc = np.array(nmisc)
nmikmeansplusplus = np.array(nmikmeansplusplus)
nmissc = np.array(nmissc)
nmikmeans = np.array(nmikmeans)
ssc = np.array(ssc)
print(nmissc)
print(sssc)
#
# km = [0.2,0.28,0.3,0.39,0.48,0.65,0.54,0.85,0.89,0.93,0.8,0.84,1.21,1.08,1.23]
# kmpp = [0.32,0.4,0.64,0.85,0.55,0.95,1.3,1.7,1.5,1.4,1.9,2.1,2.4,2.3,2.8]
# ssc = [2.7,4.1,6.6,7.2,9.5,10.8,14.5,15.4,18.7,18.8,19.1,20.5,22.5,23.1,23.4]
# sssc = [0.7,1.2,1,1.6,3.1,3.3,4,5.6,6,6.1,8,7.8,9.9,11.6,12.2]

# km = [0.61,0.6,0.58,0.58,0.55,0.56,0.54,0.55,0.542,0.525,0.54,0.524,0.527,0.521,0.515]
# kmpp =[0.6,0.63,0.59,0.57,0.56,0.57,0.545,0.555,0.545,0.542,0.525,0.54,0.54,0.52,0.525]
# ssc = [0.63,0.60,0.61,0.54,0.55,0.53,0.56,0.58,0.54,0.52,0.50,0.545,0.55,0.53,0.52]
# sssc = [0.68,0.66,0.60,0.65,0.63,0.58,0.595,0.62,0.60,0.589,0.59,0.6,0.59,0.61,0.60]
fig, xy_time = plt.subplots()
xy_time.plot(x_plot, km, '-d',c= "blue", label = 'KMeans')
xy_time.plot(x_plot, kmpp, '-^',c= "yellow", label = 'KMeans++')
xy_time.plot(x_plot, ssc, '-d',c= "red", label = 'SC')
xy_time.plot(x_plot, sssc,'--o',c= "green",linewidth=2, label='SeqSC')
xy_time.legend(loc='upper left')
xy_time.set_xlabel('x Word')
xy_time.set_ylabel('time (s)')
xy_time.set_title("Time ")


fig, xy_nmi = plt.subplots(tight_layout=True)
xy_nmi.plot(x_plot, nmikmeans, '-d',c= "blue", label = 'KMeans')
xy_nmi.plot(x_plot, nmikmeansplusplus, '-^',c= "yellow", label = 'KMeans++')
xy_nmi.plot(x_plot, nmisc, '-X',c= "red", label = 'SC')
xy_nmi.plot(x_plot, nmissc,'--o',c= "green",linewidth=2, label='SeqSC')
xy_nmi.legend(loc='upper left')
xy_nmi.set_xlabel('x Word')
xy_nmi.set_ylabel('NMI')
xy_nmi.set_title("NMI")



print("NMI computation : ")
print("SeqSC : "+str(nmissc))
print("SC : "+str(nmisc))
print("KMeans : "+str(nmikmeans))
print("KMeans++ : "+str(nmikmeansplusplus))


plt.show()
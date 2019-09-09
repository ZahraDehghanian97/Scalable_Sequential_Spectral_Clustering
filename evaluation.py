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
nmisc = []
nmikmeans = []
nmikmeansplusplus = []
repeat = 2
print("SeqSC computation")
print(" n  |  time  |  NMI")

for i in range(0,7):
    nmissc.append(0)
    nmisc.append(0)
    nmikmeansplusplus.append(0)
    nmikmeans.append(0)
    ssc.append(0)
    sssc.append(0)

for j in range(0,repeat):
    for i in range(0,7):
        n = (i+2) *50
        k = 10
        time_start = time.perf_counter()
        x, y, labels = SeqSC.guiseqsc(k,n,n,1)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y,labels)
        nmissc[i]+=(nmi)
        print(str(n)+" | "+str(elapsed)+"  |  "+str(nmi))
        ssc[i]+= (elapsed)


print("SC computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,7):
        n = (i+1) *100
        k = 10
        time_start = time.perf_counter()
        x, y, labels = SC.guisc(k,n,1)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        nmisc[i]+=(nmi)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        sssc[i]+=(elapsed)


print("KMeans computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,7):
        n = (i+1) *100
        k = 10
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeans(k,n,1)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        nmikmeans[i]+=(nmi)


print("KMeans++ computation")
print(" n  |  time  |  NMI")
for j in range(0,repeat):
    for i in range(0,7):
        n = (i+1) *100
        k = 10
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeansplusplus(k,n,1)
        elapsed = time.perf_counter() - time_start
        nmi = normalized_mutual_info_score(y, labels)
        print(str(n) + " | " + str(elapsed) + "  |  " + str(nmi))
        nmikmeansplusplus[i]+=(nmi)

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

fig, xy_time = plt.subplots()
fig, xy_nmi = plt.subplots(1, 3, tight_layout=True)
x = [1000,2000,3000,4000,5000,6000,7000]
xy_time.plot(x, ssc, '-d',c= "red", label = 'SC')
xy_time.plot(x, sssc,'-o',c= "green",linewidth=2, label='SeqSC')
xy_time.legend(loc='upper left')
xy_time.set_xlabel('x Word')
xy_time.set_ylabel('time (s)')
xy_time.set_title("Time SC - SSC")

xy_nmi[0].plot(x, nmisc, '-d',c= "red", label = 'SC')
xy_nmi[0].plot(x, nmissc,'-o',c= "green",linewidth=2, label='SeqSC')
xy_nmi[0].legend(loc='upper left')
xy_nmi[0].set_xlabel('x Word')
xy_nmi[0].set_ylabel('NMI')
xy_nmi[0].set_title("NMI SC - SSC")

xy_nmi[1].plot(x, nmikmeans, '-d',c= "blue", label = 'KMeans')
xy_nmi[1].plot(x, nmissc,'-o',c= "green",linewidth=2, label='SeqSC')
xy_nmi[1].legend(loc='upper left')
xy_nmi[1].set_xlabel('x Word')
xy_nmi[1].set_ylabel('NMI')
xy_nmi[1].set_title("NMI KM - SSC")

xy_nmi[2].plot(x, nmikmeansplusplus, '-d',c= "yellow", label = 'KMeans++')
xy_nmi[2].plot(x, nmissc,'-o',c= "green",linewidth=2, label='SeqSC')
xy_nmi[2].legend(loc='upper left')
xy_nmi[2].set_xlabel('x Word')
xy_nmi[2].set_ylabel('NMI')
xy_nmi[2].set_title("NMI KM++ - SSC")

print("NMI computation : ")
print(x)
print("SeqSC : "+str(nmissc))
print("SC : "+str(nmisc))
print("KMeans : "+str(nmikmeans))
print("KMeans++ : "+str(nmikmeansplusplus))


plt.show()
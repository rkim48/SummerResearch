import itertools
import numpy as np
import math
import matplotlib.pyplot as plt

def AAPE(x, m, delay, A):
    n = len(x)
    permutations = np.array(list(itertools.permutations(range(m))))
    # window_c holds motif number for a window i in time series input x
    window_c = np.zeros(n - delay * (m - 1))
    # c holds the weighted probability values for motif j
    c = np.zeros([len(window_c),len(permutations)])
    for i in range(n - delay * (m - 1)):
        sorted_array = np.array(np.sort(x[i:i + delay * m:delay], kind='quicksort'))
        sorted_index_array = np.array(np.argsort(x[i:i + delay * m:delay], kind='quicksort'))

        if min(abs(np.diff(sorted_array))) != 0:
            for j in range(len(permutations)):
                if abs(permutations[j] - sorted_index_array).any() == 0:
                    c[i,j] = c[i,j] + (A / m) * np.sum(abs(x[i:i + delay * m:delay])) + \
                           ((1 - A) / (m - 1)) * np.sum(abs(np.diff(x[i:i + delay * m:delay])))
                    window_c[i] = j

        else:

            u1, u2, u3 = np.unique(x[i:i + delay * m:delay], return_index=True,
                                   return_inverse=True)
            tf = 1
            for k in range(len(u1)):
                ind = np.where(u3 == k)[0]
                tf = math.factorial(len(ind)) * tf

            y = np.nan * np.ones([tf, m])

            for k in range(len(u1)):
                a = np.where(u3 == k)[0]
                pa = np.array(list(itertools.permutations(a)))
                y[:, a] = np.tile(pa, (int(math.floor(tf / math.factorial(len(a)))), 1))

            for l in range(tf):
                iv = y[l, :]

                for jj in range(len(permutations)):

                    if abs(permutations[jj] - iv).any() == 0:
                        c[i,jj] = c[i,jj] + (1 / tf) * (A / m) * np.sum(abs(x[i:i + delay * m:delay])) + \
                                ((1 - A) / (m - 1)) * np.sum(abs(np.diff(x[i:i + delay * m:delay])))
                        window_c[i] = jj
    hist = c
    c = c[np.nonzero(c)]
    p = c / np.sum(c)
    aape = -np.sum(p * np.log(p))
    # print(x)
    # print(permutations)
    # print(aape)
    # print(hist)
    # print(window_c)
    return p, aape, hist, window_c

def compute_AAMI(X,Y,m,delay,A):

    permutations = math.factorial(m)
    p = permutations
    Fxy = np.zeros([p, p])
    condEntropyYX = 0

    # vectors containing permutation motifs for all windows of data
    Fx, hX, X_weighted_prob, X_motifs = AAPE(X, m, delay, A)
    Fy, hY, Y_weighted_prob, Y_motifs = AAPE(Y, m, delay, A)
    print hX,hY
    print X_weighted_prob
    print X_motifs
    windows = len(X_motifs)
    for i in range(windows):
        # motif type for window i
        X_motif_i = int(X_motifs[i])
        Y_motif_i = int(Y_motifs[i])
        # for m = 3, Fxy will be 6x6 array

        Fxy[Y_motif_i, X_motif_i] += np.mean([X_weighted_prob[i,X_motif_i], Y_weighted_prob[i,Y_motif_i]])
    print Fxy

    for i in range(permutations):
        # conditional entropy of Y given X = i
        YgivenXi = Fxy[:, i]
        YgivenXi = YgivenXi[np.nonzero(YgivenXi)]
        if len(YgivenXi) > 0:
            condProbYXi = YgivenXi / np.sum(YgivenXi)
            condEntropyYX -= Fx[i] * np.dot(condProbYXi, np.log(condProbYXi))

    Ixy = hY - condEntropyYX
    print "Ixy", Ixy
    return Ixy

# x = np.array([1,2,3,4,5,4,3,2,1])
y = np.array([1,4,3,2,4,1,5,4,3])
x = np.array([1,10,3,2,20,1,24,22,5])
plt.subplot(1,2,1)
plt.scatter(np.linspace(0,x.shape[0]+1,x.shape[0]),y)
plt.subplot(1,2,2)
plt.scatter(np.linspace(0,x.shape[0]+1,x.shape[0]),x)
plt.show()
# print x
print y
#
# AAPE(x, 3, 1, 0.5)
compute_AAMI(x,y,3,1,0.5)
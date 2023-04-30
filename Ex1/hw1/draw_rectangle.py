import time
import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import cholesky

import igraph as ig


imgname = 'cross'
input_path = f'data/imgs/{imgname}.jpg'
rect = tuple(map(int, open(f"data/bboxes/{imgname}.txt", "r").read().split(' ')))
img = cv2.imread(input_path)
x, y, w, h = rect
h-=1
w-=1
img[y, x:w] = 0
img[y:h, x] = 0
img[y:h,w] = 0
img[h, x:w] = 0
cv2.imshow('my img', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
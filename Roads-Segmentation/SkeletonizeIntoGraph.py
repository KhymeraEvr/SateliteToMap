import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.io import imread
from skimage import morphology, filters
import sknw
from PIL import Image
import numpy as np

# open and skeletonize
img = imread("10828720_15.tif")
binary = img > filters.threshold_otsu(img)
ske = skeletonize(binary).astype(np.uint16)

# build graph from skeleton
graph = sknw.build_sknw(ske)

g = graph.adj;

for pts in graph.nodes.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    for x,y in pts[1]['pts']:
        xD = abs(x - 1244 );
        yD = abs(y - 1009);
        if (xD < 20) & (yD < 20):
            print("x :" + str(x))
            print("y :" + str(y))

print("searched")

# draw image
plt.imshow(binary, cmap='gray')

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

# title and show
plt.title('Build Graph')
plt.show()
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.io import imread
from skimage import morphology, filters
import sknw
from PIL import Image
import numpy as np
import networkx as nx

inputImagePath = 'Graphs/input/'
poitSearchTreshhold = 100

class Graph:
    binaryImg = None;
    def GetGraphFromImage(self, imageFileName):
        # open and skeletonize
        filePath = inputImagePath + imageFileName;
        img = imread(filePath)
        binary = img > filters.threshold_otsu(img)
        self.binaryImg = binary
        ske = skeletonize(binary).astype(np.uint16)

        # build graph from skeleton
        graph = sknw.build_sknw(ske)

        return graph

    def FindNodeByXY(self, graph, X, Y):
        results = []
        for pts in graph.nodes.items():
            for x,y in pts[1]['pts']:
                xD = abs(x - X);
                yD = abs(y - Y);
                if (xD < poitSearchTreshhold) & (yD < poitSearchTreshhold):
                    results.append(pts)
                    # print("x :" + str(x))
                    # print("y :" + str(y))
        return results

    def FindNodeById(self, graph, nodeId):
        for pts in graph.nodes.items():
            if pts[0] == nodeId:
                return pts

    def FindPath(self, graph, nodeA,nodeB):
        path = nx.shortest_path(graph, nodeA, nodeB);
        return path;

    def DrawGraph(self, graph):
        # draw image
        plt.imshow(self.binaryImg, cmap='gray')

        # draw edges by pts
        for (s,e) in graph.edges():
           ps = graph[s][e]['pts']
           plt.plot(ps[:,1], ps[:,0], 'green' )

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:,1], ps[:,0], 'r.')

        # title and show
        plt.title('Build Graph')
        plt.show()

    def DrawPath(self, graph, path):
        plt.imshow(self.binaryImg, cmap='gray')

        # draw edges by pts
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')
        pathNodes = []
        for pathNd in path:
            pathPoint = self.FindNodeById(g, pathNd)
            pathNodes.append(pathPoint[1])
        ps2 = np.array([pathNodes[i]['o'] for i in range(len(pathNodes))])
        plt.plot(ps2[:, 1], ps2[:, 0], 'blue')


        # title and show
        plt.title('Build Graph')
        plt.show()

        # draw image
        # plt.imshow(binary, cmap='gray')

        # draw edges by pts
        # for (s,e) in graph.edges():
        #    ps = graph[s][e]['pts']
        #    plt.plot(ps[:,1], ps[:,0], 'green')

        # draw node by o
        # nodes = graph.nodes()
        # ps = np.array([nodes[i]['o'] for i in nodes])
        # plt.plot(ps[:,1], ps[:,0], 'r.')

        # title and show
        # plt.title('Build Graph')
        # plt.show()

serv = Graph()
g = serv.GetGraphFromImage("22978990_15.tif")
node1 = serv.FindNodeByXY(g, 1422, 1373)[0]
node2 = serv.FindNodeByXY(g, 400, 400)[0]
path = serv.FindPath(g,node1[0],node2[0])
#serv.DrawGraph(g);
serv.DrawPath(g,path)

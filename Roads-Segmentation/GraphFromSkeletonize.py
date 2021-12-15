import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.io import imread
from skimage import morphology, filters
import sknw
from PIL import Image
import numpy as np
import networkx as nx

inputImagePath = 'Graphs/input/'
Closing2Folder = 'SegmentRemoval/Closing2/'
outputImagePath = 'Graphs/output/'
poitSearchTreshhold = 10

class Point:
    x = 0
    y = 0

    def __init__(self, X, Y):
        self.x = X
        self.y = Y


class Graph:
    binaryImg = None;

    def Show(self, img):
        fig, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest')
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def ShowSkeleton(self, image, ske):
        # display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                 sharex=True, sharey=True)

        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(ske, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()

    def SaveSkeleton(self, image, ske, fileName):
        # display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                 sharex=True, sharey=True)

        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(ske, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.savefig('Graphs/Skeletons/' + fileName)

    def GetGraphFromImage(self, imageFileName):
        # open and skeletonize
        filePath = Closing2Folder + imageFileName;
        img24bit = imread(filePath)
        img =  np.array(Image.fromarray(img24bit).quantize(colors=2, method=2))
        # self.Show(img)

        binary = img > filters.threshold_otsu(img,100)

        #self.Show(img);

        self.binaryImg = binary
        ske = skeletonize(binary).astype(np.uint16)

        #self.ShowSkeleton(img,ske)
        self.SaveSkeleton(img, ske, imageFileName)

        # build graph from skeleton
        graph = sknw.build_sknw(ske)

        return graph

    def FindNodeByXY(self, graph, X, Y):
        results = []
        print("Searching X: " + str(X) + "  Y: " + str(Y))
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
        path = nx.shortest_path(graph, nodeA, nodeB, weight='weight');
        path_length = nx.shortest_path_length(graph, nodeA, nodeB, weight='weight');
        result = {
            "path": path,
            "len": path_length
        }
        return result;

    def DrawGraph(self, graph, fileName):
        # draw image
        #plt.imshow(self.binaryImg, cmap='gray')

        # draw edges by pts
        for (s,e) in graph.edges():
           ps = graph[s][e]['pts']
           plt.plot(ps[:,1], ps[:,0], 'green' )

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:,1], ps[:,0], 'r.')

        #for nd in nodes:
         #   if (nd == 276) | (nd == 74):
         ##       nodePoint = self.FindNodeById(g, nd)
         #       plt.annotate(nd, (nodePoint[1]['o'][1], nodePoint[1]['o'][0]), color='white')

        # title and show
        plt.title('Build Graph')
        plt.savefig('Graphs/Graphs/' + fileName)

        #plt.show()

    def DrawPath(self, graph, path, fileName):
        fig, ax = plt.subplots()
        plt.imshow(self.binaryImg, cmap='gray')

        # draw edges by pts
        for (s, e) in graph.edges():
         #   print('s ' + str(s) + ' e ' + str(e))
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        nodes = graph.nodes()

        #ps = np.array([nodes[i]['o'] for i in nodes])
        #plt.plot(ps[:, 1], ps[:, 0], 'r.')

        for nd in nodes:
           if (nd == path[0]) | (nd == path[-1]):
               nodePoint = self.FindNodeById(graph, nd)
               plt.plot(nodePoint[1]['o'][1], nodePoint[1]['o'][0], 'r.')
               #plt.annotate(nd, (nodePoint[1]['o'][1], nodePoint[1]['o'][0]), color='white')

        pathNodes = []
        for pathNd in path:
            pathPoint = self.FindNodeById(graph, pathNd)
            pathNodes.append(pathPoint[1])
            plt.plot(pathPoint[1]['o'][1], pathPoint[1]['o'][0], 'blue')
            # plt.annotate(pathPoint[0], (pathPoint[1]['o'][1], pathPoint[1]['o'][0]), color='white')

        ps2 = np.array([pathNodes[i]['o'] for i in range(len(pathNodes))])
        plt.plot(ps2[:, 1], ps2[:, 0], 'blue')


        # title and show
        #plt.savefig('Graphs/Paths/' + fileName)
        #plt.show()
        # Image from plot
        ax.axis('off')
        fig.tight_layout(pad=0)

        # To remove the huge white borders
        ax.margins(0)

        fig.canvas.draw()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Cropped image of above dimension
        # (It will not change original image)
        im = Image.fromarray(image_from_plot)
        im1 = im.crop((80, 0, 560, 480))

        im1.save('../ImageGraph/ImageGraph/wwwroot/' + fileName)
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

    def GetPathImage(self, fileName, cors1x, cors1y, cors2x, cors2y):
        g = self.GetGraphFromImage(fileName)
        self.DrawGraph(g, fileName);

        node1 = self.FindNodeByXY(g, cors1x, cors1y)[0]
        print("point1 = " + str(node1))
        node2 = self.FindNodeByXY(g, cors2x, cors2y)[0]
        print("point2 = " + str(node2))
        pathRes = self.FindPath(g, node1[0], node2[0])
        self.DrawPath(g, pathRes["path"], fileName)
        return fileName

    def GetPathImage(self, fileName, points):
        g = self.GetGraphFromImage(fileName)
        self.DrawGraph(g, fileName);
        resultFile = 'Graphs/Graphs/' + fileName;

        path = [];
        pathLength = 0;
        iters = len(points) - 1;
        for i in range(iters):
            j = i;
            node1 = self.FindNodeByXY(g, points[j].x, points[j].y)[0]
            print("point1 = " + str(node1))
            node2 = self.FindNodeByXY(g, points[1+j].x, points[1+j].y)[0]
            print("point2 = " + str(node2))
            pathRes = self.FindPath(g, node1[0], node2[0])
            path = path + pathRes["path"]
            pathLength = pathRes["len"];

        self.DrawPath(g, path, fileName)

        result = {
            "img": fileName,
            "len": pathLength
        }
        return result


#points = [ Point(36, 27), Point(229, 220), Point(229, 37) ]

#serv = Graph()
#fileName = "34_pred.png";
#g = serv.GetGraphFromImage(fileName)
#serv.DrawGraph(g,fileName);


#serv.GetPathImage(fileName, points);
#node1 = serv.FindNodeByXY(g, 36, 27)[0]
#print("point1 = " +str(node1))
#node2 = serv.FindNodeByXY(g, 229, 220)[0]
#print("point2 = " +str(node2))
#path = serv.FindPath(g,node1[0],node2[0])
#serv.DrawPath(g,path, fileName)

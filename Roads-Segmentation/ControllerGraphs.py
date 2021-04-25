import flask
import DNNPredict
import GraphFromSkeletonize
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/segment/<filename>', methods=['GET'])
def segmentRoads(filename):
    classifier = DNNPredict.DnnClassifier()
    resultFilename = classifier.predict(filename)

    return resultFilename

@app.route('/shortestPath/<filename>', methods=['POST'])
def shortestPath(filename):
    serv = GraphFromSkeletonize.Graph()
    result = serv.GetGraphFromImage(filename)
    print(result)
    return result

app.run()
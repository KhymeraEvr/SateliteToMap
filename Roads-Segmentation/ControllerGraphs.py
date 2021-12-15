import flask
import U_net_predict_given
import GraphFromSkeletonize
import SegmentsRemove
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

resizedImageSize = 256
def normalizeCors(cor, maxV):
    corF = float(cor)
    maxF = float(maxV)
    res = 256 / maxF * corF
    return res


@app.route('/shortestPath', methods=['POST'])
def shortestPath():
    data = flask.request.json
    filename = data['fileName']
    predictedName = U_net_predict_given.Predict(filename)
    processed = SegmentsRemove.RemoveSegments(predictedName)
    serv = GraphFromSkeletonize.Graph()
    maxY = data['maxY']
    maxX = data['maxX']
    cors = data['cors']

    print(data);
    points = [];
    for cor in cors:
        print(cor['X'])
        normX = normalizeCors(cor['Y'], maxX)
        normY = normalizeCors(cor['X'], maxY)
        point = GraphFromSkeletonize.Point(normX, normY)
        points.append(point)

    result = serv.GetPathImage(processed, points)

    print(result)
    return result

app.run()
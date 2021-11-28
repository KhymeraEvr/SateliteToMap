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
    res = 256 / maxV * corF
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
    points = data['cors']
    for point in points:
        point.X = normalizeCors(pointX, maxY)
    y1 = normalizeCors(data['y1'], maxY)
    y2 = normalizeCors(data['y2'], maxY)
    x1 = normalizeCors(data['x1'], maxX)
    x2 = normalizeCors(data['x2'], maxX)
    result = serv.GetPathImage(processed, y1, x1, y2, x2)

    print(result)
    return result

app.run()
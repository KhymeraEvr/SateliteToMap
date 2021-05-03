import flask
import U_net_predict_given
import GraphFromSkeletonize
import SegmentsRemove
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/shortestPath', methods=['POST'])
def shortestPath():
    data = flask.request.json
    filename = data['fileName']
    predictedName = U_net_predict_given.Predict(filename)
    processed = SegmentsRemove.RemoveSegments(predictedName)
    serv = GraphFromSkeletonize.Graph()
    result = serv.GetPathImage(processed, int(data['y1']), int(data['x1']), int(data['y2']), int(data['x2']))

    print(result)
    return result

app.run()
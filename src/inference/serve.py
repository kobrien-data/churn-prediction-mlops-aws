import os
from flask import Flask, request, Response
from inference import model_fn, input_fn, predict_fn, output_fn

app = Flask(__name__)
model = model_fn(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))


@app.route('/ping', methods=['GET'])
def ping():
    return Response('', status=200)


@app.route('/invocations', methods=['POST'])
def invoke():
    data = input_fn(request.data.decode('utf-8'), request.content_type)
    prediction = predict_fn(data, model)
    result, content_type = output_fn(prediction, request.accept_mimetypes.best)
    return Response(result, content_type=content_type)

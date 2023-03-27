from flask import Flask, jsonify, make_response
from run import *
from datasets import *
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
current_iteration = 20
BASE_PATH = ""


def add_Access(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'GET, POST, OPTIONS')
    response.headers.add("X-Powered-By", ' 3.2.1')
    response.headers.add("Content-Type", "application/json;charset=utf-8")
    response.headers.add('Access-Control-Allow-Methods',
                         'DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization')
    return response


@app.route("/init", methods=["GET"])
def first_run():
    global current_iteration
    try:
        if current_iteration == 0:
            print(0)
        else:
            path = BASE_PATH
            sample_data = pd.read_csv(os.path.join(
                path, 'sample_data.csv')).to_dict()
            epoch_Data = pd.read_csv(os.path.join(
                path, 'epoch_Data.csv')).to_dict()
            WSI_Data = pd.read_csv(os.path.join(
                path, 'WSI_Data.csv')).to_dict()
            confusion_Data = pd.read_csv(
                os.path.join(path, 'confusion.csv')).to_dict()
            bk_data = pd.read_csv(os.path.join(
                path, 'bk_data.csv')).to_dict()

            response = make_response(jsonify({
                'iteration': current_iteration,
                'sample_data': sample_data,
                'epoch_Data': epoch_Data,
                'WSI_Data': WSI_Data,
                'confusion_Data': confusion_Data,
                'bk_data': bk_data,
            }))
    except Exception as e:
        print("[Exception]:", e)
        response = make_response(jsonify({"load_status": 500}))

    return add_Access(response)


if __name__ == '__main__':
    app.run(use_reloader=False)

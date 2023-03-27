from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import os.path as osp
from run import *
from datasets import *
import pandas as pd
import os
import io
import base64
from PIL import Image


app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False

# # read config
config = read_config()
#
current_iteration = 20
BASE_PATH = "F:\TVCG_Project\TVCG"

data_name = "peso"  # peso / hubmap


def add_Access(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'GET, POST, OPTIONS')
    response.headers.add("X-Powered-By", ' 3.2.1')
    response.headers.add("Content-Type", "application/json;charset=utf-8")
    response.headers.add('Access-Control-Allow-Methods',
                         'DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization')
    return response


def normalize_df(df, key=[]):
    smooth = 1e-5
    for k in key:
        k_l = np.array(df[k].to_list())
        k_max = np.max(k_l)
        k_min = np.min(k_l[k_l != 0])
        df[k] = (k_l-k_min)/(k_max-k_min+smooth)
    return df


@app.route("/init", methods=["GET"])
def first_run():
    global current_iteration
    global config
    global logger
    model = None
    dataLoader = None
    try:
        if current_iteration == 0:
            print(0)
        else:
            print("loading data")
            if data_name == "peso":
                path = os.path.join(BASE_PATH, "Visual/public/data/save_data")

            elif data_name == "hubmap":
                path = os.path.join(
                    BASE_PATH, "hubmapVisual/public/data/save_data")

            sample_data = pd.read_csv(os.path.join(
                path, 'sample_data.csv')).to_dict()
            epoch_Data = pd.read_csv(os.path.join(
                path, 'epoch_Data.csv')).to_dict()
            WSI_Data = normalize_df(pd.read_csv(os.path.join(path, 'WSI_Data.csv')), key=[
                                    'o2us', 'fines']).to_dict()
            # WSI_Data=pd.read_csv(os.path.join(path, 'WSI_Data.csv')).to_dict()

            confusion_Data = pd.read_csv(os.path.join(
                path, 'confusion.csv')).to_dict() if data_name == "peso" else None
            bk_data = pd.read_csv(os.path.join(
                path, 'bk_data.csv')).to_dict() if data_name == "peso" else None

            response = make_response(jsonify({
                'load_status': 200,
                'dataset': data_name,
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

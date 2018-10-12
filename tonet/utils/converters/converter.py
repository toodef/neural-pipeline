import json
import os

import onnx
import torch

import numpy as np

from onnx_caffe2.backend import Caffe2Backend

import cv2
from torchvision import transforms

from tonet.tonet import DataProcessor, FileStructManager, Model, StateManager

result_dir = r"D:\\sunflowers_configs"
config_dir = r"Z:\nn_projects\sunflowers\workdir\1"

# img = torch.autograd.Variable(torch.randn(1, 3, 256, 256)).cpu()

img = cv2.imread(r"D:\datasets\sunflower\train\images\PRODIMEX_KURSK_4_03082017_PhotoLeft_g201b20086_f004_169.jpg")
img = cv2.resize(img, (256, 256))
cv2.imshow("img", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = img.copy() / 255
# img = cv2.normalize(img, 1.0, -128, dtype=cv2.CV_32FC3)
img2 = np.swapaxes(img2, -1, 0)
img2 = np.array([img2])

# normalize = transforms.Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[1 / (.0167 * 255)] * 3)
# torch_img = torch.from_numpy(np.moveaxis(img2.copy().astype(np.float32) / 255., -1, 0))
# torch_img = normalize(torch_img)

torch_img = torch.from_numpy(img2.copy()).detach().cpu().float()

onnx_weights = os.path.join(result_dir, "onnx", "torch_out.proto")
caffe2_init_weights = os.path.join(result_dir, "caffe2", "init.pb")
caffe2_predict_weights = os.path.join(result_dir, "caffe2", "predict.pb")
tf_weights = os.path.join(result_dir, "tf", "graph.meta")

os.makedirs(os.path.join(result_dir, "caffe2"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "onnx"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "tf"), exist_ok=True)


def export_to_onnx(pytorch_model):
    print("Export to ONNX")
    torch.onnx.export(pytorch_model, torch_img, onnx_weights)
    print("Done")


def predict_image_caffe2():
    import caffe2.python.workspace as w

    print("Start predict image")
    with open(caffe2_init_weights, 'rb') as f:
        init_net = f.read()
    with open(caffe2_predict_weights, 'rb') as f:
        predict_net = f.read()

    global img
    local_img = img.copy().astype(np.float32)
    local_img = local_img - 128
    local_img = np.reshape(np.swapaxes(local_img, -1, 0), (1, 3, 256, 256))

    # tmp = np.swapaxes(128 + local_img[0], 0, -1).astype(np.uint8)
    # cv2.imshow("dfdfdf", tmp)
    # cv2.waitKey()

    res = w.Predictor(init_net, predict_net).run([local_img])

    res = np.reshape(res, (256, 256, 1))
    res = np.swapaxes(res, 0, 1)
    # res = np.roll(res, 0, axis=0)
    min = np.min(res)
    max = np.max(res)
    res = (255 * (res - min) / (max - min)).astype(np.uint8)
    cv2.imshow("predict_caffe2", res)
    print("Done")


def convert_to_caffe2(onnx_model):
    print("Start convert to caffe2")
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph, device="CPU")

    print("  Write init ...")
    with open(caffe2_init_weights, "wb") as f:
        f.write(init_net.SerializeToString())
    print("  Done")
    print("  Write predict ...")
    with open(caffe2_predict_weights, "wb") as f:
        f.write(predict_net.SerializeToString())
    print("Done")


def convert_to_tf(onnx_model):
    import onnx_tf.backend as tf_backend
    import tensorflow as tf

    backend = tf_backend.prepare(onnx_model)
    with tf.Session() as sess:
        meta_graph_def = tf.train.export_meta_graph(filename=tf_weights)


if __name__ == "__main__":
    # print("Start converting model with PyTorch version: ", torch.__version__)
    #
    # with open(os.path.join(config_dir, "config.json"), 'r') as f:
    #     cfg = json.load(f)
    #
    # file_struct_manager = FileStructManager(os.path.join(config_dir, "config.json"))
    # cfg['data_processor']['start_from'] = 'continue'
    # # data_processor = DataProcessor(cfg['data_processor'], file_sruct_manager, classes_num=1)
    #
    # model = Model(cfg['data_processor'], file_struct_manager, classes_num=1, is_cuda=False)  # data_processor.model().cpu().eval()
    #
    # state_manager = StateManager(file_struct_manager)
    # state_manager.unpack()
    # model.load_weights(state_manager.get_files()['weights_file'])
    # state_manager.clear_files()
    #
    # data = torch.from_numpy(np.moveaxis(img.astype(np.float32) / 255., -1, 0))
    # normalize = transforms.Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[1 / (.0167 * 255)] * 3)
    # data = normalize(data)
    # data = torch.autograd.Variable(data.unsqueeze(0).contiguous().cpu())
    #
    # res = model(data).cpu().data.numpy()
    # res = np.squeeze((res - np.min(res)) / (np.max(res) - np.min(res)))
    # # res = np.swapaxes(res, 0, -1)
    # cv2.imshow("predict_pytorch", res)
    #
    # export_to_onnx(model)
    # onnx_model = onnx.load(onnx_weights)
    # onnx.checker.check_model(onnx_model)
    # convert_to_caffe2(onnx_model)

    # predict_image_caffe2()
    # cv2.waitKey()

    convert_to_mxnet()

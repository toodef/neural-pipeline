import os

result_dir = r"D:\\sunflowers_configs"
onnx_file = os.path.join(result_dir, "onnx", "torch_out.proto")


def convert_to_mxnet():
    from mxnet.contrib import onnx as onnx_mxnet

    # Import the ONNX model into MXNet's symbolic interface
    sym, arg, aux = onnx_mxnet.import_model(onnx_file)
    print("Loaded {}".format(onnx_file))
    print(sym.get_internals())


if __name__ == "__main__":
    convert_to_mxnet()

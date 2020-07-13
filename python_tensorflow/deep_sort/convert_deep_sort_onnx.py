import os
import sys
from subprocess import call


def convert(graph_path: str, out_name):
    """
    https://github.com/onnx/tensorflow-onnx/issues/509
    use latest opset 10 it solves the problem
    """
    # input_name = "net/images:0"
    # output_name = "net/features:0"
    inputs = "images:0"
    outputs = "features:0"
    tf2onnx_name = "tf2onnx.convert"

    call(f"{sys.executable} -m {tf2onnx_name} --graphdef {graph_path} --opset 10 --output {out_name}"
         f" --inputs {inputs} --outputs {outputs}", env=os.environ)


if __name__ == '__main__':
    export = '../models/deep_sort/saved/deep_sort.onnx'
    graph_pb = '../models/deep_sort/mars-small128.pb'
    convert(graph_pb, export)

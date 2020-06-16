import argparse

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


def save(graph_pb, export_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # INFO: name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name='')
        g = tf.get_default_graph()

        # INFO: if name is added the input/output should be prefixed like:
        #       name=net => net/images:0 & net/features:0
        inp = tf.get_default_graph().get_tensor_by_name("images:0")
        out = tf.get_default_graph().get_tensor_by_name("features:0")

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"in": inp}, {"out": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save(as_text=True)


def save2(graph_pb, export_dir):
    from tensorpack import get_default_sess_config

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    proto = get_default_sess_config()

    tf.import_graph_def(graph_def, name='net')

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for n in nodes:
        print(f"node name : {n}")

    # inp = tf.get_default_graph().get_tensor_by_name("net/images:0")
    # out = tf.get_default_graph().get_tensor_by_name("net/features:0")

    with tf.Session(config=proto, graph=tf.get_default_graph()) as sess:

        nodes = [n.name for n in sess.graph.as_graph_def().node]
        for n in nodes:
            print(f"node name : {n}")

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            ["net/features"]
            # output_node_names []
        )
        out_pb = export_dir + "/freeze.pb"
        out_pbtxt = export_dir + "/freeze.pbtxt"

        with tf.gfile.GFile(str(out_pb), "wb") as f:
            f.write(frozen_graph.SerializeToString())
        tf.io.write_graph(frozen_graph, '', out_pbtxt)


if __name__ == '__main__':
    export_dir = './saved'
    graph_pb = '../models/deep_sort/mars-small128.pb'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="path to frozen pb file")
    parser.add_argument('--output', help="Folder to save")
    args = parser.parse_args()
    if args.input is not None and args.output:
        # save(args.input, args.output)
        save2(graph_pb, export_dir)
    else:
        print(f"Usage adapt_opencv.py.py --input 'path_to_bp' --output './saved'")

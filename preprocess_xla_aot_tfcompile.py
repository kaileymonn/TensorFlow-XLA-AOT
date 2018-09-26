import argparse
import copy
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2


def node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]

def extract_graph_summary(graph_def):
    """Extracts useful information from the graph and returns them."""
    name_to_input_name = {}  # Keyed by the dest node name.
    name_to_node = {}  # Keyed by node name.

	# Keeps track of node sequences. It is important to still output the
	# operations in the original order.
    name_to_seq_num = {}  # Keyed by node name.
    seq = 0
    for node in graph_def.node:
        n = node_name(node.name)
        name_to_node[n] = node
        name_to_input_name[n] = [node_name(x) for x in node.input]
        name_to_seq_num[n] = seq
        seq += 1
    return name_to_input_name, name_to_node, name_to_seq_num

parser = argparse.ArgumentParser(description='Preprocessing for XLA AOT compilation. Issues handled: Cast, ResizeBilinear')
parser.add_argument('--input-graph', help='Required argument: input_model.pb')
parser.add_argument('--output-graph', default='model_with_dequantized_casts.pb', help='Output protobuf')
args = parser.parse_args()

model = args.input_graph
dest = args.output_graph

with tf.gfile.GFile(model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

new_model = graph_pb2.GraphDef()
with tf.Session(graph=graph) as sess:
    name_to_input_name, name_to_node, name_to_seq_num = extract_graph_summary(sess.graph_def)
    for n in sess.graph_def.node:
        if n.op == 'Cast' and n.attr['SrcT'].type not in [3, 9, 22, 23]:
            nn = node_def_pb2.NodeDef()
            parent = node_def_pb2.NodeDef()
            nn.CopyFrom(n)
            nn.attr['SrcT'].type = 3
            parent.CopyFrom(name_to_node[n.input[0]])
            if 'dtype' in parent.attr:
                parent.attr['dtype'].type = 3
            elif 'T' in parent.attr:
                parent.attr['T'].type = 3
            else:
                import code
                code.interact(local=locals())
            new_model.node.remove(name_to_node[n.input[0]])
            new_model.node.extend([parent])
            new_model.node.extend([nn])
        elif n.op == 'ResizeBilinear' and n.attr['align_corners'].b == False:
            # import code
            # code.interact(local=locals())
            nn = node_def_pb2.NodeDef()
            nn.CopyFrom(n)
            nn.attr['align_corners'].b = True
            new_model.node.extend([nn])
        else:
            nn = node_def_pb2.NodeDef()
            nn.CopyFrom(n)
            new_model.node.extend([nn])
        
with tf.gfile.GFile(dest, "wb+") as f:
    f.write(new_model.SerializeToString())

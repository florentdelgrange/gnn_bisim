from typing import Callable, Union, List, Tuple
import tensorflow as tf

from .base import Base
from .utils import adjIdTensor, adjTensor, glorot,\
                   sparseDropout, sparseTensorFromMatrix,\
                   zeros

import scipy.sparse as sp
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


IDFAC = 0.5872  # id_factor default value for all GNNs

class GNNBisimLayer(tfkl.Layer):
    def __init__(
        self,
        graph,                          # graph to get the norm'd adj. matrix
        in_dim,                         # input dimension (previous label dimension)
        out_dim,                        # output dimension (next label dimension)
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        id_factor=IDFAC,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.id_factor = id_factor
        self.adjacency_tensor = adjTensor(graph)
        self.n_vertices = self.adjacency_tensor.get_shape()[0]
        self.weight_matrix = glorot(
            shape=[in_dim, out_dim], name="weights")
        self.bias_vector1 = zeros([self.n_vertices, out_dim], name="bias1")
        self.bias_vector2 = zeros([self.n_vertices, out_dim], name="bias2")
        self.activation = activation
    
    def call(self, inputs, *args, **kwargs):
        inputs = sparseTensorFromMatrix(inputs)
        breakpoint()

        pI = sparseTensorFromMatrix(self.id_factor * sp.eye(self.n_vertices))
        pIL_prev = tf.sparse.sparse_dense_matmul(pI, inputs)
        x = tf.sparse.sparse_dense_matmul(pIL_prev, self.weight_matrix)

        Z = tf.sparse.sparse_dense_matmul(self.adjacency_tensor, inputs)
        Z = tf.sparse.sparse_dense_matmul(Z, self.weight_matrix)
        Z = self.activation(tf.sparse.add(Z, self.bias_vector1))
         
        new_label = tf.sparse.add(x, Z)
        new_label = tf.sparse.add(new_label, self.bias_vector2)
        new_label = self.activation(new_label)

        return new_label

class GNNBisim(tfk.Sequential):
    def __init__(
        self,
        graph,                          # graph to get the norm'd adj. matrix
        input_dim: int,
        output_dim: int,
        hidden_layers: Union[List[int], Tuple[int, ...]] = (16, 16),
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
        id_factor=IDFAC,
        *args,
        **kwargs
    ):
        hidden_layers = list(hidden_layers)
        layers = []
        _in_dim = input_dim
        for _out_dim in hidden_layers + [output_dim]:
            layers.append(
                GNNBisimLayer(
                    graph,
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    id_factor=id_factor,
                    activation=activation))
            _in_dim = _out_dim
        super().__init__(layers=layers, *args, **kwargs)

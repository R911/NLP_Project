import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model


def create_model(node_size, embedding_size, proximity_order='second'):
    vertex_i = Input(shape=(1,))
    vertex_j = Input(shape=(1,))

    first_order_embeddings = Embedding(node_size, embedding_size, name='first_order_embeddings')
    second_order_embeddings = Embedding(node_size, embedding_size, name='second_order_embeddings')
    context_embeddings = Embedding(node_size, embedding_size, name='context_embeddings')

    vertex_i_foe = first_order_embeddings(vertex_i)
    vertex_j_foe = first_order_embeddings(vertex_j)

    vertex_i_soe = second_order_embeddings(vertex_i)
    vertex_j_ce = context_embeddings(vertex_j)

    first = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keepdims=False), name='first_order')([vertex_i_foe, vertex_j_foe])
    second = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keepdims=False), name='second_order')([vertex_i_soe, vertex_j_ce])

    if proximity_order == 'first':
        output_list = [first]
    elif proximity_order == 'second':
        output_list = [second]
    else:
        output_list = [first, second]

    model = Model(inputs=[vertex_i, vertex_j], outputs=output_list)

    return model, {'first': first_order_embeddings, 'second': second_order_embeddings}


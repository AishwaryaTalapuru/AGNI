import tensorflow as tf
import random

def random_graph(num_layers=5, input_shape=(32, 32)):
    x = tf.random.normal([1] + list(input_shape))
    out = x
    for i in range(num_layers):
        op_type = random.choice(['matmul', 'conv', 'add', 'relu', 'mul'])
        if op_type == 'matmul':
            w = tf.random.normal([out.shape[-1], out.shape[-1]])
            out = tf.matmul(out, w)
        elif op_type == 'conv':
            if len(out.shape) == 2:
                out = tf.expand_dims(out, -1)
            filters = tf.random.normal([3, 3, out.shape[-1], out.shape[-1]])
            out = tf.nn.conv2d(out, filters, strides=1, padding='SAME')
        elif op_type == 'add':
            out = out + tf.random.normal(out.shape)
        elif op_type == 'relu':
            out = tf.nn.relu(out)
        elif op_type == 'mul':
            out = out * tf.random.normal(out.shape)
    return out

def generate_graph_batch(batch_size=100):
    return [random_graph(num_layers=random.randint(3, 10),
                         input_shape=(random.randint(8, 128), random.randint(8, 128)))
            for _ in range(batch_size)]

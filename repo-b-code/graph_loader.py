import tensorflow as tf

class GraphLoader:
    @staticmethod
    def load_from_function(tf_function, sample_input):
        concrete_func = tf_function.get_concrete_function(sample_input)
        return concrete_func.graph

    @staticmethod
    def load_from_saved_model(path):
        loaded = tf.saved_model.load(path)
        return loaded.signatures['serving_default'].graph

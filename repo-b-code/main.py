import tensorflow as tf
from graph_loader import GraphLoader
from agni_core import AgniCore
from passes.shape_inference import ShapeInferencePass
from passes.dead_code_elimination import DeadCodeEliminationPass

# Define the Keras model
def example_func():
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(128,)),
        tf.keras.layers.Dense(10)
    ])
    return keras_model

if __name__ == "__main__":
    # Prepare sample input matching the model's input shape
    sample_input = tf.random.normal([8, 128])  # batch size 8, feature size 128

    # Wrap the Keras model call in a tf.function to create a computation graph
    keras_model = example_func()
    @tf.function(jit_compile=True)
    def model_fn(x):
        return keras_model(x)

    # Load computation graph from tf.function
    graph = GraphLoader.load_from_function(model_fn, sample_input)

    # Initialize AGNI core
    agni = AgniCore(graph)

    print("AGNI ", agni)

    # Register analysis/transformation passes
    agni.register_pass(ShapeInferencePass())
    agni.register_pass(DeadCodeEliminationPass())

    # Run the AGNI pipeline
    agni.run()

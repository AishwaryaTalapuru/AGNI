import tensorflow as tf

class TFGraphBuilder:
    """
    A utility class to convert a TensorFlow model or function into an XLA-compiled computation graph,
    and run inference on it.
    """

    def __init__(self, model=None, model_fn=None, input_signature=None):
        """
        Initialize the graph builder.

        Args:
            model (tf.keras.Model, optional): A TensorFlow Keras model.
            model_fn (callable, optional): A TensorFlow function that takes tf.Tensor input(s).
            input_signature (list of tf.TensorSpec, optional): Input signature for tracing the function.
                Required if using model_fn.
        """
        if model is None and model_fn is None:
            raise ValueError("Either 'model' or 'model_fn' must be provided.")

        self.model = model
        self.model_fn = model_fn
        self.input_signature = input_signature

        # Create the tf.function with XLA JIT compilation enabled
        self._create_compiled_function()

    def _create_compiled_function(self):
        """
        Create a tf.function with jit_compile=True from the model or model_fn.
        """
        if self.model is not None:
            # Wrap the Keras model call in a tf.function with XLA
            @tf.function(jit_compile=True)
            def compiled_fn(x):
                return self.model(x)
            self.compiled_fn = compiled_fn

        else:
            # Use the provided model_fn and input_signature to create a compiled tf.function
            if self.input_signature is None:
                raise ValueError("input_signature must be provided when using model_fn.")

            self.compiled_fn = tf.function(self.model_fn,
                                           input_signature=self.input_signature,
                                           jit_compile=True)

    def get_concrete_function(self, sample_input):
        """
        Get the concrete function (computation graph) for a given sample input.

        Args:
            sample_input (tf.Tensor or tuple/list of tf.Tensor): Sample input tensor(s) matching the input_signature.

        Returns:
            tf.function.ConcreteFunction: The concrete function representing the compiled graph.
        """
        return self.compiled_fn.get_concrete_function(sample_input)

    def run(self, input_tensor):
        """
        Run inference using the compiled function.

        Args:
            input_tensor (tf.Tensor or tuple/list of tf.Tensor): Input tensor(s) to the model.

        Returns:
            tf.Tensor or tuple of tf.Tensor: The output tensor(s) from the model.
        """
        return self.compiled_fn(input_tensor)


if __name__ == "__main__":
    # Example usage

    # Example 1: Using a Keras model
    print("Example 1: Keras Model")

    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(128,)),
        tf.keras.layers.Dense(10)
    ])

    builder1 = TFGraphBuilder(model=keras_model)
    sample_input1 = tf.random.normal([16, 128])
    concrete_fn1 = builder1.get_concrete_function(sample_input1)

    output_str = "Concrete Computation graph (Keras model):\n"
    output_str += str(concrete_fn1.graph.as_graph_def()) + "\n"
    print("Successfully printed Concrete function graph (Keras model)")
    #print(concrete_fn1.graph.as_graph_def())

    output1 = builder1.run(sample_input1)
    output_str += f"Output shape: {output1.shape}\n"
    # Write the string to a text file
    with open("input_comp_graph.txt", "w") as f:
        f.write(output_str)
    print("Successfully printed Output shape in the text file", output1.shape)
    print()


    """
    # Example 2: Using a custom model function with input signature
    print("Example 2: Custom model_fn")

    def custom_model_fn(x):
        w1 = tf.Variable(tf.random.normal([128, 64]))
        w2 = tf.Variable(tf.random.normal([64, 10]))
        x = tf.nn.relu(tf.matmul(x, w1))
        return tf.matmul(x, w2)

    input_sig = [tf.TensorSpec(shape=[None, 128], dtype=tf.float32)]

    builder2 = TFGraphBuilder(model_fn=custom_model_fn, input_signature=input_sig)
    sample_input2 = tf.random.normal([8, 128])
    concrete_fn2 = builder2.get_concrete_function(sample_input2)
    print("Concrete function graph (custom model_fn):")
    print(concrete_fn2.graph.as_graph_def())

    output2 = builder2.run(sample_input2)
    print("Output shape:", output2.shape)
    """

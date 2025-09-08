from .pass_base import PassBase

class ShapeInferencePass(PassBase):
    def run(self, graph):
        print("Shape Inference Results:")
        for op in graph.get_operations():
            print(f"Op: {op.name}")
            for out in op.outputs:
                print(f"  Output: {out.name}, Shape: {out.shape}")

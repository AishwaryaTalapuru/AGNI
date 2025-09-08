from .pass_base import PassBase

class DeadCodeEliminationPass(PassBase):
    def run(self, graph):
        used_ops = set()
        for op in graph.get_operations():
            for inp in op.inputs:
                used_ops.add(inp.op)
        all_ops = set(graph.get_operations())
        unused_ops = all_ops - used_ops
        print("Unused operations:")
        for op in unused_ops:
            print(f"  {op.name}")

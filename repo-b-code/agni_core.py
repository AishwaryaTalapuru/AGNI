class AgniCore:
    def __init__(self, tf_graph):
        self.graph = tf_graph
        self.passes = []

    def register_pass(self, pass_obj):
        self.passes.append(pass_obj)

    def run(self):
        print("Hii")
        for p in self.passes:
            p.run(self.graph)

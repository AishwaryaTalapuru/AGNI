class PassBase:
    def run(self, graph):
        raise NotImplementedError("Pass must implement the run() method.")

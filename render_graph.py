from graphviz import Source

dot_file_path = "graph.dot"

# Read the DOT content from the file
with open(dot_file_path, "r") as f:
    dot_data = f.read()

graph = Source(dot_data)
graph.render("computation_graph", format="png", cleanup=True)
graph.view()

import sys
from graphviz import Source

# Check if the user provided a file path
if len(sys.argv) < 2:
    print("Usage: python render_graph.py <path_to_dot_file>")
    sys.exit(1)

dot_file_path = sys.argv[1]

# Read the DOT content from the file
with open(dot_file_path, "r") as f:
    dot_data = f.read()

# Create and render the graph
graph = Source(dot_data)
graph.render("computation_graph", format="png", cleanup=True)
graph.view()


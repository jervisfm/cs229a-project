
import feed_forward_neural_network
from ann_visualizer.visualize import ann_viz;

model = feed_forward_neural_network.model()

print("Visualizing ...")
ann_viz(model, title="Feed forward neural network")
print("Done")
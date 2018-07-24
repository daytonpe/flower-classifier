import numpy as np
hidden_layers = 5
node_list = np.linspace(1920, 102, hidden_layers+2).astype(int).tolist()
for idx, val in enumerate(node_list[1:-2]):
    print(node_list[idx+1], node_list[idx+2])

from devolution_network import DevolutionNetwork
import torch
import json
import os


def save_model(net, path):
    os.makedirs(path, exist_ok=True)
    torch.save(net.state_dict(), path + "/" + "model.pth")
    with open(path + "/" + 'model.json', 'w') as fp:
        json.dump(
            {
                "input_size": net.input_size,
                "hidden_layer_sizes": net.hidden_layer_sizes,
                "output_size": net.output_size}
            , fp)


def load_model(path):
    with open(path + "/" + "model.json", 'r') as fp:
        data = json.load(fp)
        net = DevolutionNetwork(**data)
        net.load_state_dict(torch.load(path + "/" + "model.pth"))
    return net

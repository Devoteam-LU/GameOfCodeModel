from devolution_network import DevolutionNetwork
import torch
import json
import os
import pickle


def save_model(net, path):
    os.makedirs(path, exist_ok=True)
    torch.save(net.state_dict(), path + "/" + "model.pth")
    with open(path + "/" + 'model.json', 'w') as fp:
        json.dump(
            {
                "input_size": net.input_size,
                "hidden_layer_sizes": net.hidden_layer_sizes,
                "output_size": net.output_size,
                "mean": net.mean.tolist(),
                "std": net.std.tolist(),
                # "transform": net.transform
            }
            , fp)
    # with open(path + "/" + 'model.json', 'w') as f:
    #     pickle.dump(str({
    #         "input_size": net.input_size,
    #         "hidden_layer_sizes": net.hidden_layer_sizes,
    #         "output_size": net.output_size,
    #         "transform": net.transform
    #     })
    #         , f)


def load_model(path):
    with open(path + "/" + "model.json", 'r') as fp:
        # with open('mypickle.pickle') as fp:
        #     data = pickle.load(fp)
        data = json.load(fp)
        net = DevolutionNetwork(**data)
        net.load_state_dict(torch.load(path + "/" + "model.pth"))
    return net

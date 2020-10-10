from model import DevolutionNetwork
import torch
import json
import os

def load_model(model_dir):
    model_info_path = os.path.join(model_dir, 'model.json')
    with open(model_info_path, 'r') as fp:
        data = json.load(fp)
        net = DevolutionNetwork(**data)
        model_path = os.path.join(model_dir, 'model.pth')
        net.load_state_dict(torch.load(model_path))
    return net

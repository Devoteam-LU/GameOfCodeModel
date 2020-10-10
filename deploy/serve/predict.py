import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from utils import load_model


# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    model = load_model(model_dir)
    
    model.to(device).eval()
    print("Done loading model.")
    
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return prediction_output


def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model.
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The variable `out_label` should be a rounded value, either 1 or 0
    out = model(data)
    out_np = out.cpu().detach().numpy()
    out_label = out_np.round()

    return out_label

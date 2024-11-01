import pandas as pd
import json
import numpy as np
import torch

def features_to_df(data : dict, type="global") -> pd.DataFrame:

    assert type in ["global", "participant", "contribution"], "type must be one of 'global', 'participant', or 'contribution'"

    
    if type == "global":

        df = pd.DataFrame()
    
        for key, value in data.items():
            print(key)
            print(value)

            flattened = pd.json_normalize(value)
            flattened.index = [key]
            df = pd.concat([df, flattened])
            
        return df
    

    if type == "participant":
        records = []
        index = []

        for key, value in data.items():
            for participant, participant_data in value.items():
                flattened = pd.json_normalize(participant_data)
                records.append(flattened)
                index.append((key, participant))

        df = pd.concat(records, ignore_index=True)
        df.index = pd.MultiIndex.from_tuples(index, names=["key", "participant"])
        return df
        
    if type == "contribution":
        records = []
        index = []

        for key, entry in data.items():
            for i, contribution in enumerate(entry):
                flattened = pd.json_normalize(contribution)
                records.append(flattened)
                index.append((key, i))

        df = pd.concat(records, ignore_index=True)
        df.index = pd.MultiIndex.from_tuples(index, names=["key", "index"])
        return df   
                
               


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


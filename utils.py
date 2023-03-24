import torch
import torch.nn.functional as F
from deep_rescnn import Deep_ResCNN

def import_model(task:str, device):
    assert task in ['ar','mi'], 'unknown task, support only task [ar,mi]'
    
    # Initiate required num_classes of the task
    num_classes = {
        'ar' : 5, 
        'mi' : 1
        }
    
    # Setup architecture
    model = Deep_ResCNN(num_classes = num_classes[task]).to(device)
    
    # Load state_dict
    model.load_state_dict(
        torch.load(f'{task}_classification/best_deep_rescnn_model_state_dict.pt',
                   map_location=device)    
        )
    
    return model

def preprocess(X,fix_seq_len = 187):
    """
    Fill to make the input tensor satisfy with tensor shape (1,1,187) bu padding or truncate (last)
    """
    seq_len = X.shape[2]

    if seq_len < fix_seq_len:
        diff = fix_seq_len - seq_len
        X = torch.nn.functional.pad(X, (0, diff)) #pad the sample

    elif seq_len > fix_seq_len:
        X = X[:, :, :fix_seq_len] # truncate-last on the sample
    
    assert X.shape == (1,1,187)
    return X

def predict_ar(model, data):
    # Initiate the map_dct
    ar_map_dct = {
        0:'normal',
        1:'Supra-ventricular premature',
        2:'Ventricular escape',
        3:'Fusion of ventricular and normal',
        4:'unclassifiable'
        }
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        pred = ar_map_dct[pred.item()]
        prob = prob.max()
    
    output = {
        'Class' :pred,
        'Probability' : prob.item()
            }
    
    return output

def predict_mi(model, data):
    # Initiate the map_dct
    mi_map_dct = {
        0:'normal',
        1:'myocardial infarction'
        }
    
    #Initiate the threshold (optimal)
    threshold = 0.46949878334999084
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        prob = torch.sigmoid(output)
        pred = torch.where(prob > threshold, 1, 0)
        pred = mi_map_dct[pred.item()]
        
        # prob for negative will be ..
        if pred == 'normal':
            prob = 1-prob
    
    output = {
        'Class' :pred,
        'Probability' : prob.item()
            }
    
    return output
import torch
from utils import import_model, preprocess, predict_ar, predict_mi
from flask import Flask, render_template, redirect, url_for, request
import os
import ast 

# Setup things necessary for inferencing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AR_MODEL = import_model('ar',DEVICE) # Arhythmic
MI_MODEL = import_model('mi',DEVICE) # myocardial infarction

# Setup Flask things
app = Flask(__name__, template_folder='templates')

@app.route('/')
def hello():
    return render_template('hello.html')

@app.route('/predict_<string:task>',methods = ['GET','POST'])
def predict(task):
    """
    For Production Deployment, No User Interface to speed up inferencing
    
    The key as a predictor must be 'beat_input with csv file'
    """
    
    # Validate the available task
    if task not in ['ar','mi']:
        return 'Unknown task, make sure to choose /predict_ar or /predict_mi'
    
    # Select the right model for the task
    if task == 'ar':
        predict_func = predict_ar
        model = AR_MODEL
    elif task == 'mi':
        predict_func = predict_mi
        model = MI_MODEL
        
    # Retreive the input either POST or GET
    if request.method == 'POST':
        input_json = request.get_json()
        if 'beat_input' not in input_json:
            return 'Make sure you have key as "beat_input" in your JSON payload'
        beat_input = input_json['beat_input']
    else:
        beat_input = request.args.get('beat_input')
    
    if beat_input is None:
        return 'Make sure you have key as "beat_input"'
    
    # Convert string to list '[1,2]' -> [1,2]
    beat_input = ast.literal_eval(beat_input) 
    if not isinstance(beat_input, list):
        return 'Make sure the input is list. Eg. /predict_ar?beat_input=[0.4,0.38,0.6]'
    
    # Transform beat_input which originally was a list into tensor with the desired shape
    beat_input = torch.tensor(beat_input,dtype=torch.float32).reshape(1, 1, len(beat_input))
    # Preprocess
    beat_input = preprocess(beat_input) # In case the sequence length is not 187 #truncate or pad
    beat_input = beat_input.to(DEVICE)  
    # Inferencing for both ar and mi  
    output = predict_func(model,beat_input)
    
    return output # change 3 to any integer 

if __name__ == '__main__':
    app.run()

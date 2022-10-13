from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


# Select HuggingFace Model to use.
model_ckpt = "dslim/bert-base-NER"

# Specify the path to download the model. 
cache_dir = "models"

# Download the model.
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, cache_dir=cache_dir)
model = AutoModelForTokenClassification.from_pretrained(model_ckpt, cache_dir=cache_dir)

# Use the pipeline abstraction class to initialize the model.
ner_model = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')

# Define the FastAPI app.
app = FastAPI()

# Define the Request Body Schema.
class Request_body_scheme(BaseModel):
    text: Optional[str]
 
# Create a route for prediction.
@app.post('/predict')
def ner(payload: Request_body_scheme):
    if payload.text:
        # Make prediction using the model.
        prediction = ner_model(payload.text)
        # Create the response body.
        response = {
            'word': [dict_['word'] for dict_ in prediction],
            'entity': [dict_['entity_group'] for dict_ in prediction]
            }
        return response
    return {'message': "No text provided"}
    
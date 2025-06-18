from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import spacy
from huggingface_hub import snapshot_download
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

app = FastAPI()

# Download and load spaCy model
model_path = snapshot_download("kalyanram3600/en_resume_ner_pipeline")
nlp = spacy.load(model_path)

# Request schema
class ResumeText(BaseModel):
    text: str

# Response schema
class Entity(BaseModel):
    label: str
    text: str

@app.get("/")
def home():
    return {"message": "Resume NER API is running."}

@app.post("/predict", response_model=List[Entity])
def extract_entities(data: ResumeText):
    doc = nlp(data.text)
    results = [{"label": ent.label_, "text": ent.text} for ent in doc.ents]
    return results

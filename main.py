from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import spacy
from huggingface_hub import snapshot_download
import os

# Prevent symlink issues on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

app = FastAPI()

# Download and load the spaCy model
model_path = snapshot_download(
    "kalyanram3600/en_resume_ner_pipeline",
    local_dir="en_resume_model",
    local_dir_use_symlinks=False
)
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
    return [{"label": ent.label_, "text": ent.text} for ent in doc.ents]

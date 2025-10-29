#!pip install fastapi uvicorn pymongo pillow numpy sentence-transformers requests

# 🧩 IMAGE SIMILARITY DEMO — EDUCATIONAL PLACEMATS (CLIP MODEL)
# ===============================================================

from pymongo import MongoClient
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import requests, numpy as np
from bson import ObjectId

app = FastAPI()
client = MongoClient("mongodb+srv://anna_db_user:6zxpOoyMUqnpxrBS@similaritysearch.xblvd4g.mongodb.net/")
db = client["similaritysearch"]
model = SentenceTransformer("sentence-transformers/all-MiniCLIP-L6-v2")

@app.post("/generate_embedding")
def generate_embedding(data: dict):
    img_url = data["thumbnail"]
    image = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB").resize((512, 512))
    emb = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
    db.images.update_one({"_id": ObjectId(data["_id"])}, {"$set": {"embedding": emb.tolist()}})
    return {"message": "embedding added"}

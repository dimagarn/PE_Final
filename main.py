from transformers import pipeline
# import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)


@app.get("/")
async def root():
    return {'model': 'Text Classifier'}

# def classify(sentences):
#     classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

#     model_outputs = classifier(sentences)
#     return model_outputs

# def print_predictions(predictions):
#     for elem in predictions:
#         for item in elem:
#             st.write(f"{item.get('label').title()}: {item.get('score')}")

# st.title("Классификатор настроения текста")
# sentences = st.text_input('Предложение')
# classified = classify(sentences)
# if (st.button('Отправить') and sentences) or sentences:
#     st.write("Настроение предложения: ")
#     print_predictions(classified)

@app.get("/predict/")
def predict():
   return classifier('I like dogs!')[0]


@app.post("/predict/")
def predict(item: Item):
    """Text Classifier"""
    return classifier(item.text)[0]
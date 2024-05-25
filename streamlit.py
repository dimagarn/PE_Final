from transformers import pipeline
import streamlit as st


def classify(sentences):
    classifier = pipeline(task="text-classification",
                          model="SamLowe/roberta-base-go_emotions",
                          top_k=None)
    model_outputs = classifier(sentences)
    return model_outputs


def print_predictions(predictions):
    for elem in predictions:
        for item in elem:
            st.write(f"{item.get('label').title()}: {item.get('score')}")


st.title("Классификатор настроения текста")
sentences = st.text_input('Предложение')
classified = classify(sentences)
if (st.button('Отправить') and sentences) or sentences:
    st.write("Настроение предложения: ")
    print_predictions(classified)
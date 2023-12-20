from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["Hello everyone, and today i want to introduce you my new amazing product!"]

model_outputs = classifier(sentences)
print(model_outputs[0])

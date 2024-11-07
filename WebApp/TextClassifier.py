from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("./best_model")
model = AutoModelForSequenceClassification.from_pretrained("./best_model")

class_names = {
    0: "Household",
    1: "Books",
    2: "Clothing and Accessories",
    3: "Electronics"
}

st.title("E-commerce Text Classification")
user_input = st.text_input("Enter the text to classify")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

if st.button("Classify"):
    if user_input:
        prediction = classifier(user_input)
        predicted_class_idx = int(prediction[0]['label'].split('_')[1])
        predicted_class = class_names.get(predicted_class_idx, "Unknown")
        st.write("Predicted class:", predicted_class)
    else:
        st.write("Please enter text for classification.")

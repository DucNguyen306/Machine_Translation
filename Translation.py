import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import Convert.py
dich = st.text_input('Nhập vào câu cần dịch')
click = st.button('Translate')
if click:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = T5ForConditionalGeneration.from_pretrained("model/model")
    tokenizer = T5Tokenizer.from_pretrained("model/tokenizer")
    model.to(device)
    tokenized_text = tokenizer.encode(dich, return_tensors="pt").to(device)
    model.eval()
    summary_ids = model.generate(
        tokenized_text,
        max_length=128,
        num_beams=5,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(output)
    st.write(output)

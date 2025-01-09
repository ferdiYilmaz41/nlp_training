from transformers import AutoTokenizer, AutoModel
import torch

#model ve tokenizer yükleme
model_name= "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)

model=AutoModel.from_pretrained(model_name)

#Örnek metin
text= "I am a student at the university, transformers are amazing for nlp tasks."

#Metni tokenize etme
inputs=tokenizer(text, return_tensors="pt")

#Metin temsili oluşturma
with torch.no_grad():
    output=model(**inputs)

#Çıktıyı yazdırma
last_hidden_state=output.last_hidden_state
first_token_embedding = last_hidden_state[0,0,:].numpy()
print("First Token: ",first_token_embedding)

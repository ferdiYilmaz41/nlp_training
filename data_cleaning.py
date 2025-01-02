#Metinlerdeki fazla boşlukları temizliyoruz

text= "Naber,   moruk  nasılsın sadsd   d sa   ds."
text.split()
cleaned_text=" ".join(text.split())
print(cleaned_text)

# %% büyük harfleri küçüğe çevirme
text=" DSNFJSKDFNSDJK34"
cleaned_text=text.lower()

# %% noktalama işaretlerini kaldırma
text1="Naber,   **???moruk  nasılsın sadsd   d sa   ds."
import string
cleaned_text2=text1.translate(str.maketrans("", "", string.punctuation))
print(cleaned_text2)

# %% özel karakterlerinden kurtulma

import re

text2="Naber,   **???moruk  nasılsın sadsd   d sa   ds."
cleaned_text3= re.sub(r"[^A-Za-z0-9\s]", "", text2)

#%% yazım hatalarını düzeltme

from textblob import TextBlob 
text2="whats ap brothur how arı uofgbfgdh"
cleaned_text4= str(TextBlob(text2).correct())
#%% html izlerini silme
from bs4 import BeautifulSoup
text5="<div> Deneme yapıyoruz so</>nuçta yani"
cleaned_text5=BeautifulSoup(text5,"html.parser").get_text()
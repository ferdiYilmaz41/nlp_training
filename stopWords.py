from nltk.corpus import stopwords
import nltk

# Download the stop words list
nltk.download('stopwords')

# Get the list of stop words
stop_words = set(stopwords.words('english'))

# Example text
text = "This is a sample sentence, showing off the stop words filtration."

# Tokenize the text
words = text.split()

# Filter and print the stop words
stop_words_in_text = [word for word in words if word.lower() in stop_words]
print("Stop words in text:", stop_words_in_text)

# Custom stop words
custom_stop_words = {'this', 'is', 'a', 'sample'}

# Filter and print the custom stop words
custom_stop_words_in_text = [word for word in words if word.lower() in custom_stop_words]
print("Custom stop words in text:", custom_stop_words_in_text)
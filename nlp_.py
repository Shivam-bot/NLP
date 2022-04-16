from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
example_set = "Hi, This is Shivam. Looking for gr8 change which will bring by me. If you want that change then get it by yourself."
stop_words = set(stopwords.words('english'))
# print(sent_tokenize(example_set))
# print(word_tokenize(example_set))
print(stop_words)




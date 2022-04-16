from nltk import word_tokenize, PunktSentenceTokenizer, pos_tag
from nltk.corpus import state_union
import nltk
train_example = state_union.raw("2005-GWBUSH.txt")
test_example = state_union.raw("2006-GWBUSH.txt")
example_train = PunktSentenceTokenizer(train_example)
custom_tokenize = example_train.tokenize(test_example)


def process_content(data):
    try:
        for i in data:
            words = word_tokenize(i)
            tagged_data = pos_tag(words)
            chunk_gram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>}"""
            chunkparser = nltk.RegexpParser(chunk_gram)
            chunked = chunkparser.parse(tagged_data)
            chunked.draw()

    except Exception as e:
        print(str(e))


process_content(custom_tokenize)
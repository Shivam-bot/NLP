import nltk
import random
from nltk.corpus import movie_reviews
import pickle


document = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(document)

all_words = [w.lower() for w in movie_reviews.words()]
all_words = nltk.FreqDist(all_words)

word_feature = list(all_words.keys())[:3000]


def words_feature(document):
    words = set(document)
    feature = {}
    for w in word_feature:
        feature[w] = (w in words)

    return feature

featureset = [(words_feature(rev),category) for rev, category in document]

trainig_set = featureset[:1900]
testing_set = featureset[1900:]

# classifier = nltk.NaiveBayesClassifier.train(trainig_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print('Naive Bayes classifier accuracy is', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(16)

# save_classifier = open("naivebayes.pickle",'wb')
# pickle.dump(classifier,save_classifier)
# save_classifier.close()



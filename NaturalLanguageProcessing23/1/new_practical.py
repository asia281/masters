"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


#################################
# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the
            corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    for text in corpus:
        corpus_words.extend([word for word in text])
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted(list(set(['Ala', 'END', 'START', 'i', 'kot', 'lubic', 'miec', 'pies'])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert (num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(
    ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(
    str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the
                same as the ordering of the words given by the distinct_words
                function.
            word2Ind (dict): dictionary that maps word to index
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ------------------
    # Write your implementation here.
    for i, word in enumerate(words):
        word2Ind[word] = i

    d = dict()
    M = np.zeros((num_words, num_words))

    for text in corpus:
        for cnt, word in enumerate(text):
            tokens = text[cnt + 1:cnt + 1 + window_size]
            for token in tokens:
                key = tuple(sorted([token, word]))
                if key not in d:
                    d[key] = 0
                d[key] += 1

    for key, value in d.items():
        x = word2Ind[key[0]]
        y = word2Ind[key[1]]
        M[x, y] = M[y, x] = value

    # ------------------

    return M, word2Ind


# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(
    test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array([
    [0., 0., 2., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [0., 1., 0., 1., 0., 1., 1., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 1., 0., 0., 0., 0.]
])

word2Ind_ans = {
    'Ala': 0, 'END': 1, 'START': 2, 'i': 3, 'kot': 4, 'lubic': 5, 'miec': 6,
    'pies': 7}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans,
                                                                                                     word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape,
                                                                                                          M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError(
                "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1,
                                                                                                                  idx2,
                                                                                                                  w1,
                                                                                                                  w2,
                                                                                                                  student,
                                                                                                                  correct))

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.

    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=13)
    M_reduced = svd.fit_transform(M)

    # ------------------

    print("Done.")
    return M_reduced


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 8), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 8)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words, filename=None):
    """ Plot in a scatterplot the embeddings of the words specified
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
    """

    # ------------------
    # Write your implementation here.
    x_val = []
    y_val = []

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        x_val.append(x)
        y_val.append(y)

    fig, ax = plt.subplots()
    ax.scatter(x_val, y_val)

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        ax.annotate(word, (x, y))

    if filename:
        plt.savefig(filename)
    # ------------------#


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below.
# ---------------------

print("-" * 80)
print("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {
    'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

print("-" * 80)


#################################
# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt', cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(), categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words, "unnormalized.png")


def plot_normalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]  # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, "normalized.png")


pl_corpus = read_corpus_pl()
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]

plot_normalized(pl_corpus, words)
plot_unnormalized(pl_corpus, words)

# What clusters together in 2-dimensional embedding space? 

# For normalized plot, we got two clusters: one for word "sztuka" and another one for the rest of the words.
# For unnormalized we got three clusters: first for "sztuka", second for "literatura"
# the third one for "spiewaczka", "obywatel" and "poeta". It makes a lot of sense, becase
# in the third cluster we have words related to the human. 

# What doesn’t cluster together that you might think should have?

# It's counterintuitive that for both plots "sztuka" and "literatura" are quite far away from 
# each other. 


# TruncatedSVD returns U × S, so we normalize the returned vectors in the second plot, so that all the vectors will appear around the unit circle. Is normalization necessary?

# The normalization removes information from vectors and plot looks worse than the one unnormalized.
# The clusters are worse in normalized.png.

#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory.
# Note: This might take several minutes.
wv_from_bin_pl = KeyedVectors.load("../word2vec_100_3_polish.bin")


# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind


# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)
M, word2Ind = get_matrix_of_vectors(wv_from_bin_pl, words)
M_reduced = reduce_to_k_dim(M, k=2)

words = ["sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]
plot_embeddings(M_reduced, word2Ind, words, "reduced.png")

# What clusters together in 2-dimensional embedding space? 
# It's not obvious question, we don't see any exact clusters, vector words 
# are quite far away from another one. I guess that again, we can cluster poeta and obywatel,
# probably with śpiewaczka, but śpiewaczka is further from poeta and obywatel.

# What doesn’t cluster together that you might think should have?
# Because I don't see exact clusters, I think that literally nothing clustered
# properly.

# How is the plot different from the one generated earlier from the co-occurrence matrix?
# Meaning of words is more visible, the closest word to spiewaczka is poeta,
# and the closest one to sztuka is literatura. 
# It wasn't visible in the previous plots.


#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.
def polysemous_pl(word: str):
    polysemous = wv_from_bin_pl.most_similar(word)
    for key, similarity in polysemous:
        print(key, similarity)

polysemous_pl("stówa")
# słowa 0.6893048286437988
# cent 0.6367954015731812
# słowo 0.6246823072433472
# stówka 0.6103435158729553
# słówko 0.608944833278656
# pens 0.5825462937355042
# tów 0.5744858980178833
# wers 0.573552668094635
# centym 0.5726915597915649
# komunał 0.5709105730056763

# polysemous_pl("babka")
# Tried "babka" ("woman" or a type of cake, but in the top 10 there were only words related to "woman")

polysemous_pl("staw")
# stawa 0.840743362903595
# jeziorko 0.760422945022583
# miednica 0.6800762414932251
# rzepka 0.6719450354576111
# niecka 0.6687400341033936
# strumyk 0.6671221852302551
# koryto 0.6663556694984436
# sadzawka 0.6657816767692566
# jezioro 0.6656040549278259
# wzgórek 0.6646289229393005
# ------------------


# Many words didn't work, because they are usually used in onluy one context (such as babka),
# or are not very popular in polish language.

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

def synonyms_antonyms_pl(w1, w2, w3):
    w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
    w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

    print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
    print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

synonyms_antonyms_pl(w1 = "radosny", w2 = "pogodny", w3 = "smutny")
# Synonyms radosny, pogodny have cosine distance: 0.3306429386138916
# Antonyms radosny, smutny have cosine distance: 0.3478999137878418

synonyms_antonyms_pl(w1 = "cyfrowy", w2 = "elektroniczny", w3 = "analogowy")
# Synonyms cyfrowy, elektroniczny have cosine distance: 0.21015697717666626
# Antonyms cyfrowy, analogowy have cosine distance: 0.24363106489181519

synonyms_antonyms_pl(w1 = "blyszczacy", w2 = "lsniacy", w3 = "matowy")
# Synonyms blyszczacy, lsniacy have cosine distance: 0.4929726719856262
# Antonyms blyszczacy, matowy have cosine distance: 0.860868439078331

synonyms_antonyms_pl(w1 = "silny", w2 = "mocny", w3 = "slaby")
# Synonyms silny, mocny have cosine distance: 0.23193776607513428
# Antonyms silny, slaby have cosine distance: 0.18360120058059692

# silny appears in a quite similar context to slaby,
# and probably in less similar context to mocny,
# that may be the reason why the distance between silny and slaby is smaller 

#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
# kobieta:mezczyzna :: syn : (corka)
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))
# [('córka', 0.6928777098655701),
#  ('dziecko', 0.6763085722923279),
#  ('matka', 0.6552439332008362),
#  ('żona', 0.6547046899795532),
#  ('siostra', 0.6358523368835449),
#  ('mąż', 0.6058387160301208),
#  ('dziewczę', 0.6008315086364746),
#  ('rodzic', 0.5781418681144714),
#  ('ojciec', 0.5779308676719666),
#  ('rodzeństwo', 0.5768202543258667)]


# nauczyciel:lekarz :: szpital : (szkola)
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["szpital", "nauczyciel"], negative=["lekarz"]))

# ('szkoła', 0.7905541658401489),
# ('sierociniec', 0.7628465294837952),
# ('przedszkole', 0.730900228023529),
# ('konwikt', 0.7190622687339783),
# ('seminarium', 0.7073177099227905),
# ('gimnazjum', 0.7053801417350769),
# ('ochronka', 0.6808328628540039),
# ('żłobek', 0.6739992499351501),
# ('szkółka', 0.6706633567810059)]

# kobieta : mezczyzna :: lekarz : (lekarka)
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["lekarz", "kobieta"], negative=["mezczyzna"]))

# [('pielęgniarka', 0.652389645576477),
#  ('lekarka', 0.6262717247009277),
#  ('osoba', 0.6237451434135437),
#  ('chirurg', 0.6050892472267151),
#  ('psychiatra', 0.5909774303436279),
#  ('pacjent', 0.5898064374923706),
#  ('ginekolog', 0.5640532970428467),
#  ('akuszerka', 0.5627631545066833),
#  ('dziewczę', 0.5559123754501343),
#  ('logopeda', 0.5546327829360962)]

# For some reason "pielegniarka" is top1, even though from our intuition "lekarka" 
# should be at the first place. It may be due to bias, but other examples suggest
# it's just a coincidence.

#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.

# mezczyzna:szef :: kobieta : (szefowa?)
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["szef", "kobieta"], negative=["mezczyzna"]))

# [('własika', 0.5678122639656067),
#  ('agent', 0.5483713150024414),
#  ('oficer', 0.5411549210548401),
#  ('esperów', 0.5383270978927612),
#  ('interpol', 0.5367037653923035),
#  ('antyterrorystyczny', 0.5327680110931396),
#  ('komisarz', 0.5326411128044128),
#  ('europolu', 0.5274547338485718),
#  ('bnd', 0.5271410346031189),
#  ('pracownik', 0.5215375423431396)]

# We don't have any term related to "szefowa" in the top 10, it means that
# "szefowa" doesn't appear in the similar context as "szef".

# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['szef', 'kobieta'], negative=['mezczyzna']))

# [('własika', 0.5678122639656067),
#  ('agent', 0.5483713150024414),
#  ('oficer', 0.5411549210548401),
#  ('esperów', 0.5383270978927612),
#  ('interpol', 0.5367037653923035),
#  ('antyterrorystyczny', 0.5327680110931396),
#  ('komisarz', 0.5326411128044128),
#  ('europolu', 0.5274547338485718),
#  ('bnd', 0.5271410346031189),
#  ('pracownik', 0.5215375423431396)]

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mezczyzna', 'prezes'], negative=['kobieta']))

# [('wiceprezes', 0.6396454572677612),
#  ('czlonkiem', 0.5929950475692749),
#  ('przewodniczący', 0.5746127963066101),
#  ('czlonek', 0.5648552179336548),
#  ('przewodniczacym', 0.5586849451065063),
#  ('wiceprzewodniczący', 0.5560489892959595),
#  ('obowiazków', 0.5549101233482361),
#  ('obowiazani', 0.5544129610061646),
#  ('dyrektor', 0.5513691306114197),
#  ('obowiazany', 0.5471130609512329)]

# In the examples from f), we don't have any word related to the female form of words "szef", "prezes",
# even though intuition tells us, this is what we should expect. 
# It's possible that in a context usually male version appears. 


#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors
# ------------------
# mezczyzna:kobieta :: zolnierz : ?
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'zolnierz'], negative=['mezczyzna']))

#[('ludzie', 0.6067951917648315),
# ('dziewczyna', 0.5997867584228516),
# ('mężczyzna', 0.5818526744842529),
# ('żołnierz', 0.5648460984230042),
# ('dziewczę', 0.5419468879699707),
# ('chłopiec', 0.5342115163803101),
# ('my', 0.5148516893386841),
# ('on', 0.5142670273780823),
# ('murzyn', 0.5128365159034729),
# ('wojownik', 0.5077145099639893)]

# I computed top10 words for kobieta + zolnierz - mezczyzna. The result consist of
# words related to "kobieta", but there's no word related to female "version"
# of zolnierz.

# h) The source of bias in word vector
# The bias is produced by word vectors in dataset that is used for training.
# Some examples are probably associated with jobs where the ration between
# number of employees of each gender is not 1:1.

#################################
# Section 3:
# English part
#################################
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


wv_from_bin = load_word2vec()

#################################
# TODO:
# Find English equivalent examples for points b) to g).
#################################

#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.

def polysemous_en(word: str):
    polysemous = wv_from_bin.most_similar(word)
    for key, similarity in polysemous:
        print(key, similarity)

polysemous_en("bark")
# barks 0.6022706031799316
# Pacific_yew_tree 0.5659282803535461
# barking 0.5538278222084045
# cork_oak_tree 0.5402646660804749
# beetles_burrow 0.5370648503303528
# barky 0.5280892252922058
# cambium 0.5276377201080322
# sapwood 0.5159558057785034
# frass 0.5140877366065979
# barked 0.5108722448348999


polysemous_en("dish")

# dishes 0.7938355803489685
# risotto 0.6626606583595276
# casserole 0.6550886034965515
# crab_rangoon 0.6336711645126343
# pasta_dish 0.6292396187782288
# lamb_biryani 0.6248414516448975
# tartare_sauce 0.6178266406059265
# pastry_crust 0.6157412528991699
# giardiniera 0.6154837012290955
# shrimp_appetizer 0.6145071387290955

# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

def synonyms_antonyms_en(w1, w2, w3):
    w1_w2_dist = wv_from_bin.distance(w1, w2)
    w1_w3_dist = wv_from_bin.distance(w1, w3)

    print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
    print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

synonyms_antonyms_en(w1 = "happy", w2 = "cheerful", w3 = "sad")
# Synonyms happy, cheerful have cosine distance: 0.6162261664867401
# Antonyms happy, sad have cosine distance: 0.46453857421875
synonyms_antonyms_en(w1 = "digital", w2 = "electronic", w3 = "analog")
# Synonyms digital, electronic have cosine distance: 0.4748317003250122
# Antonyms digital, analog have cosine distance: 0.4807000160217285
synonyms_antonyms_en(w1 = "shiny", w2 = "glossy", w3 = "mat")
# Synonyms shiny, glossy have cosine distance: 0.4870768189430237
# Antonyms shiny, mat have cosine distance: 0.8636362105607986


#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
# doctor:hospital :: teacher:school 
pprint.pprint(wv_from_bin.most_similar(
    positive=["hospital", "teacher"], negative=["doctor"]))

#[('elementary', 0.6052683591842651),
# ('school', 0.5762953758239746),
# ('teachers', 0.576140284538269),
# ('classroom', 0.5456913113594055),
# ('School', 0.5349582433700562),
# ('Elementary_School', 0.5344264507293701),
# ('Elementary', 0.5304893255233765),
# ('Teacher', 0.5241969227790833),
# ('Intermediate_School', 0.5221602320671082),
# ('Middle_School', 0.5163268446922302)]



pprint.pprint(wv_from_bin.most_similar(
    positive=["son", "woman"], negative=["man"]))
# [('daughter', 0.8796941041946411),
#  ('mother', 0.8246030211448669),
#  ('niece', 0.7625203132629395),
#  ('husband', 0.739037275314331),
#  ('granddaughter', 0.7366384863853455),
#  ('eldest_daughter', 0.7307103872299194),
#  ('daughters', 0.7241971492767334),
#  ('father', 0.7208905816078186),
#  ('sister', 0.697901725769043),
#  ('grandson', 0.6924970746040344)]


pprint.pprint(wv_from_bin.most_similar(
    positive=["boss", "woman"], negative=["man"]))

# [('bosses', 0.5522644519805908),
#  ('manageress', 0.49151360988616943),
#  ('exec', 0.45940810441970825),
#  ('Manageress', 0.4559843838214874),
#  ('receptionist', 0.4474116563796997),
#  ('Jane_Danson', 0.44480547308921814),
#  ('Fiz_Jennie_McAlpine', 0.4427576959133148),
#  ('Coronation_Street_actress', 0.44275563955307007),
#  ('supremo', 0.4409853219985962),
#  ('coworker', 0.43986251950263977)]

#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.

pprint.pprint(wv_from_bin.most_similar(
    positive=["boat", "land"], negative=["water"]))

# [('boats', 0.5301659107208252),
#  ('fishing_boat', 0.5212045311927795),
#  ('yacht', 0.508728563785553),
#  ('sailboat', 0.489263117313385),
#  ('catamaran', 0.4817970395088196),
#  ('cabin_cruiser', 0.4778779447078705),
#  ('speedboat', 0.4756109416484833),
#  ('masted_sailboat', 0.4634605646133423),
#  ('sloop', 0.4604802131652832),
#  ('trawler', 0.4592331647872925)]

pprint.pprint(wv_from_bin.most_similar(
    positive=["sky", "grass"], negative=["blue"]))

# [('vegetation', 0.45566555857658386),
#  ('treetops', 0.4393109381198883),
#  ('grasses', 0.43853452801704407),
#  ('wheat_stubble', 0.43614476919174194),
#  ('weeds', 0.4334176778793335),
#  ('overhanging_branches', 0.42577505111694336),
#  ('shrubbery', 0.42460083961486816),
#  ('leaf_litter', 0.42425423860549927),
#  ('undergrowth', 0.42417919635772705),
#  ('trees', 0.41765254735946655)]

# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'head'], negative=['men']))
    
#[('chair', 0.42928388714790344),
# ('staffer', 0.38825470209121704),
# ('Raqeeb_Abdel_Latif', 0.3881603479385376),
# ('muttered_softly', 0.3822100758552551),
# ('Patricia_Smillie_Scavelli', 0.37924641370773315),
# ('tenderly_stroking', 0.3776439428329468),
# ('severe_laceration', 0.37712281942367554),
# ('businesswoman', 0.37389394640922546),
# ('receptionist', 0.37111473083496094),
# ('suffered_puncture_wound', 0.37067875266075134)]

#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors
# ------------------

pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'solider'], negative=['men']))

# [('soldier', 0.7221205830574036),
#  ('serviceman', 0.5974410176277161),
#  ('airman', 0.5473331809043884),
#  ('Soldier', 0.5266920328140259),
#  ('girl', 0.5174804925918579),
#  ('paratrooper', 0.5063685178756714),
#  ('Army_Reservist', 0.5004420876502991),
#  ('National_Guardsman', 0.4948287010192871),
#  ('guardsman', 0.48919495940208435),
#  ('policewoman', 0.48758482933044434)]


# We see differences; in c) and d) we got very similar results, with even more visible results.
# In e) some words didn't have bias (such as man : woman ::boss : ?), even thogh in polish version they had. It means that english corpus is probably more diversed, but on the other hand have
# bunch of "glued" words that are creating a lot of noise.
# In f) we see some women present in top10, as well as businesswoman as top8,
# so it's less biased than polish version.
# In g) we can see bias very similar to the Polish version, 
# but top10 words also consists of word 'policewoman', that is a little bit related.  
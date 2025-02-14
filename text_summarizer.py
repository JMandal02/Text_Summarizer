
# networkx: Creating and manipulating complex networks (e.g., word networks)
import networkx as nx
# matplotlib.pyplot: Generating visualizations (e.g., plotting graphs)
import matplotlib.pyplot as plt
# nltk: Core Natural Language Toolkit library
import nltk
# Download the 'stopwords' dataset
nltk.download('stopwords')
# nltk.corpus.stopwords: Accessing a list of common words to remove from text
from nltk.corpus import stopwords
# nltk.tokenize: Splitting text into words and sentences
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.cluster.util: Calculating cosine distance for text similarity
from nltk.cluster.util import cosine_distance
# numpy: Numerical computing with arrays and matrices
import numpy as np
# re: Working with regular expressions for pattern matching
import re

def read_article(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        filedata = file.read()
    article = filedata.split(". ")  # Splitting the entire content
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in article]
    return sentences

"""This function is designed to calculate the **similarity** between two sentences using cosine similarity."""

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    # These lines convert the sentences to lowercase to ensure case-insensitive comparison.
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    # This line creates a list of all unique words in both sentences.
    all_words = list(set(sent1 + sent2))

    # These lines initialize two vectors to store word frequencies for each sentence.
    vactor1 = [0] * len(all_words)
    vactor2 = [0] * len(all_words)

    # This loop iterates through each word in the first sentence.
    for w in sent1:
        if w in stopwords:
            continue
        vactor1[all_words.index(w)] += 1

    # This loop does the same for the second sentence.
    for w in sent2:
        if w in stopwords:
            continue
        vactor2[all_words.index(w)] += 1

    # This line calculates and returns the cosine similarity between the two vectors.
    return 1 - cosine_distance(vactor1, vactor2)

"""This function is crucial for text summarization as it creates a similarity matrix.
    This matrix quantifies the relationships between each sentence in the text.
    
"""

def build_similarity_matrix(sentences, stop_words):

    # Initialize an empty similarity matrix with dimensions equal to the number of sentences.
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

""" This function is the core of the text summarization process."""

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_article(file_name)

    # Check if the article contains any sentences.
    if not sentences:
        print("No sentences found in the text file.")
        return []

    # Build a similarity matrix between sentences.
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

     # Sort sentences based on their PageRank scores in descending order.
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Extract the top 'top_n' ranked sentences for the summary.
    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return summarize_text

# Get the file path from the user.
file_path = input("Enter the path to the text file: ")

# Use a try-except block to handle potential errors when getting the number of sentences.
try:
    top_n = int(input("Enter the number of sentences for the summary: "))
    if top_n <= 0:
        print("Please enter a positive integer for the number of sentences.")
    else:
        summary = generate_summary(file_path, top_n)
        print("\nGenerated Summary:\n", " ".join(summary))

# Handle ValueError, which occurs if the user enters non-numeric input for 'top_n'.
except ValueError:
    print("Invalid input. Please enter a valid integer for the number of sentences.")
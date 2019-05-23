import numpy as np
'''
with open('../../vocab.txt', 'r') as file:
   vocab = file.read().split('\n')

with open('/home/nik-96/Documents/datasets/fasttext/wiki-news-300d-1M.vec', 'r') as file:
    original_word_vectors = {word_vector.split(' ')[0]: word_vector.split(' ')[1:]
                             for word_vector in file.read().split('\n')}

with open('./fasttext_word_vectors.vec', 'w') as file:
    for word in vocab:
        if word in original_word_vectors.keys():
            file.write(f"{word} {' '.join(original_word_vectors[word])}\n")

'''
with open('./fasttext_word_vectors.vec', 'r') as file:
    word_vectors = {word_vector.split(' ')[0]: np.array(word_vector.split(' ')[1:], dtype='float')
                    for word_vector in file.read().split('\n')}


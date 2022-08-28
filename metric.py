import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

INPUT_SPACE = 512

def calculate_euclidean_distance(reference_db, vector):
  return np.sqrt(np.sum((reference_db - vector) ** 2, axis=1))

generator = tf.keras.models.load_model('generator.h5')
decoder = tf.keras.models.load_model('decoder.h5')
vision = tf.keras.models.load_model('vision.h5')
vision_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

input = tf.keras.layers.Input(shape = (INPUT_SPACE,))
gen_img = tf.round((generator(input)+1) * 127.5)
output = decoder(vision(vision_preprocess( gen_img )))
evaluator = tf.keras.Model(input, output)


#Read list of words from words.txt
with open('words.txt') as file:
  words = file.readlines()
  words = [line.rstrip() for line in words]

os.environ["TFHUB_CACHE_DIR"] = "tfhub-models/"
text_encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
org_words_embeddings = text_encoder_model(words).numpy()
new_words_embeddings = evaluator.predict(org_words_embeddings)

correct = 0
for i, word in enumerate(tqdm(words)):
  #choose the word from list
  CURRENT_WORD = word
  CURRENT_WORD_EMB = org_words_embeddings[i]

  distances_org = calculate_euclidean_distance(org_words_embeddings, CURRENT_WORD_EMB)
  top_n_indices_org = distances_org.argsort()[:10]
  
  distances_new = calculate_euclidean_distance(new_words_embeddings, CURRENT_WORD_EMB)
  top_n_indices_new = distances_new.argsort()[:10]

  if top_n_indices_org[0] == top_n_indices_new[0]: correct+=1

print(f'Correct answers: {correct}, reconstruction ratio: {(correct/len(words)*100):0.2f}%')
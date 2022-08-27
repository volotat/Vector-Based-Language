import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import argparse

I_SZ = 128
R_SZ = 256
B_SZ = 10
T_SZ = 50
CHANNELS = 3

def print_to_image(text_encoder_model, generator, words, image_path, spl = 5):
  words_embeddings = text_encoder_model(words).numpy()

  predicted = generator.predict(words_embeddings)

  b_img = np.ones((B_SZ + (R_SZ + B_SZ + T_SZ) * max(1, len(predicted) // spl), B_SZ + (R_SZ + B_SZ) * spl, CHANNELS), np.uint8) * 255
  for i in range(len(predicted)):
    img = ((predicted[i] + 1) * 127.5).astype(np.uint8).reshape((I_SZ,I_SZ, CHANNELS))
    img = cv2.resize(img, (R_SZ,R_SZ), interpolation = cv2.INTER_AREA)
    
    x = i % spl
    y = i // spl
    b_img[B_SZ+y*(R_SZ+B_SZ+T_SZ): (y+1)*(R_SZ+B_SZ+T_SZ)-T_SZ, B_SZ+x*(R_SZ+B_SZ):(x+1)*(R_SZ+B_SZ)] = img

    text_origin = (2*B_SZ+x*(R_SZ+B_SZ), (y+1)*(R_SZ+B_SZ+T_SZ)-2*B_SZ)
    b_img = cv2.putText(b_img, words[i], text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1, cv2.LINE_AA)

  cv2.imwrite(image_path, cv2.cvtColor(b_img, cv2.COLOR_RGB2BGR))


def text_to_image(text_encoder_model, generator, text, image_path):
  words_embeddings = text_encoder_model([text]).numpy()

  predicted = generator.predict(words_embeddings)

  img = ((predicted[0] + 1) * 127.5).astype(np.uint8).reshape((I_SZ,I_SZ, CHANNELS))
  img = cv2.resize(img, (R_SZ,R_SZ), interpolation = cv2.INTER_AREA)

  cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-t','--texts', 
    nargs='+', 
    help='List of sentences to process', 
    required=True)

  parser.add_argument(
    '-o', '--out-img',
    type=str,
    help="Path to output image",
    default="output.png"
  )

  parser.add_argument(
    '-spl', '--spl',
    type=int,
    help="Number of samples per line in the output image",
    default=5
  )

  args = parser.parse_args()

  generator = tf.keras.models.load_model('generator.h5')

  os.environ["TFHUB_CACHE_DIR"] = "tfhub-models/"
  text_encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

  if len(args.texts) == 1:
    text_to_image(text_encoder_model, generator, args.texts[0], args.out_img)
  else:
    print_to_image(text_encoder_model, generator, args.texts, args.out_img, spl = args.spl)
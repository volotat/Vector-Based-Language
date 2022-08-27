

from copyreg import constructor
from genericpath import isdir
import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub
from datetime import datetime
from random import sample
from nltk.corpus import words
from utils import *

I_SZ = 128
R_SZ = 256
INPUT_SPACE = 512
VISION_SPACE = 1280
CHANNELS = 3
EPOCHS = 500000
batch_size = 24

output_directory = 'train_process'
if not os.path.isdir(output_directory): os.makedirs(output_directory)

test_words = ["1", "2", "3", "4", "5", 
              "6", "7", "8", "9", "0",
              "desk", "table", "chair", "sofa", "bed",
              "planet", "earth", "star", "sun", "solar system"]

#load universal-sentence-encoder
os.environ["TFHUB_CACHE_DIR"] = "tfhub-models/"
text_encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Text encoder has been loaded")

words_list = words.words()


def build_generator():
    input_layer = tf.keras.layers.Input(shape = (INPUT_SPACE,))

    layer = tf.keras.layers.Reshape((4,4,32))(input_layer) 
    
    layer = tf.keras.layers.Conv2D(512, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(512, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.UpSampling2D()(layer) # 8 x 8 x 64
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Conv2D(384, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(256, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.UpSampling2D()(layer) # 16 x 16 x 64
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Conv2D(192, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(128, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.UpSampling2D()(layer) # 32 x 32 x 64
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Conv2D(96, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(64, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.UpSampling2D()(layer) # 64 x 64 x 64
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Conv2D(48, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(32, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.UpSampling2D()(layer) # 128 x 128 x 64
    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Conv2D(24, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.Conv2D(16, 3, padding = 'same')(layer)
    layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
    layer = tf.keras.layers.BatchNormalization()(layer)
    
    layer = tf.keras.layers.Conv2D(CHANNELS, 1, activation='tanh')(layer)
    
    return tf.keras.Model(input_layer, layer, name='generator')

def build_decoder():
  input_layer = tf.keras.layers.Input(shape = (VISION_SPACE,))

  layer = tf.keras.layers.Dense(2048)(input_layer) 
  layer = tf.keras.layers.LeakyReLU(alpha=0.2)(layer) 
  
  layer = tf.keras.layers.Dense(INPUT_SPACE, activation='linear')(layer)
  
  return tf.keras.Model(input_layer, layer, name='decoder')


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, clipnorm=0.001)

if (os.path.isfile('generator.h5')):
  generator = tf.keras.models.load_model('generator.h5')
else:
  generator = build_generator()
  generator.compile(optimizer=optimizer)

if (os.path.isfile('decoder.h5')):
  decoder = tf.keras.models.load_model('decoder.h5')
else:
  decoder = build_decoder()
  decoder.compile(optimizer=optimizer)

vision_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
if (os.path.isfile('vision.h5')):
  vision = tf.keras.models.load_model('vision.h5')
else:
  vision = tf.keras.applications.MobileNetV2(
      include_top=False,
      weights="imagenet",
      input_shape= (128, 128, 3),
      pooling='avg'
  )
  vision.compile(optimizer=optimizer)

def generate_batch_data(batch_size):
  #Generate captions
  captions = []
  for i in range(batch_size * 2):
    n = 1 + np.abs(np.random.normal(0, 3)).astype(int)
    caption = ' '.join(sample(words_list, n))
    captions.append(caption)

  captions_latents = text_encoder_model(captions).numpy() 
  
  X1_data = captions_latents[batch_size:]
  X2_data = captions_latents[:batch_size]
  return X1_data, X2_data


@tf.function
def coth(x):
  return tf.cosh(x) / tf.sinh(x)

@tf.function
def smooth_round(x, w=1,h=1,a=20):
  return h*(1/2 * coth(a/2) * tf.tanh(a * ( (x/w - tf.floor(x/w))-0.5 )) + 1/2 + tf.floor(x/w))

@tf.function
def img_to_255_rgb(img, steps = 10):
  return smooth_round((img+1) * 127.5 / 255 * steps) * 255 / steps

@tf.function
def ma(x, axis = -1):
  return tf.reduce_mean(tf.abs(x), axis=axis)

@tf.function
def ms(x, axis = -1):
  return tf.reduce_mean(tf.square(x), axis=axis)

@tf.function
def evaluate_metric(X_data, training=False):
  gen1_img = generator(X_data, training)
  gen1 = vision(vision_preprocess(img_to_255_rgb(gen1_img)), training)
  eval = tf.reduce_mean(tf.abs(decoder(gen1, training) - X_data), axis=-1)
  
  return eval

@tf.function
def evaluate_loss(X1_data, X2_data, training=False):
  gen1_img = generator(X1_data, training)
  gen2_img = generator(X2_data, training)

  gen1 = vision(vision_preprocess(img_to_255_rgb(gen1_img)), training)
  gen2 = vision(vision_preprocess(img_to_255_rgb(gen2_img)), training)

  l1 = tf.abs( ma(gen1 - gen2) - ma(X1_data - X2_data) )
  l2 = 0.01 * tf.abs(2 - ma(gen1_img - gen2_img, axis = (1,2,3)))
  l3 = 0.01 * (tf.abs(2 - ma(gen1 - gen2)) + ma(gen1) + ma(gen1))
  l4 = 100 * (ms(decoder(gen1, training) - X1_data) + ms(decoder(gen2, training) - X2_data))
  l5 = 0.005 * (ms(gen1_img, axis = (1,2,3)) + ms(gen2_img, axis = (1,2,3))) #this term is not necessary. but could make images more pretty
  loss  = l1 + l2 + l3 + l4

  return loss

@tf.function
def train_on_batch(X1_data, X2_data):
  with tf.GradientTape() as tape:
    loss  = evaluate_loss(X1_data, X2_data, training=True)

  variables = generator.trainable_variables + vision.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return loss


print('Training started...')
min_loss = None
avg_loss = 0
for epoch in range(1, EPOCHS+1):
  X1_data, X2_data = generate_batch_data(batch_size)
  avg_loss += np.mean(train_on_batch(X1_data, X2_data).numpy())

  if epoch % 100 == 0:
    X1_data, X2_data = generate_batch_data(512)
    ev_loss = np.mean(evaluate_loss(X1_data, X2_data).numpy())
    ev_metric = np.mean(evaluate_metric(X1_data).numpy())
    print(epoch, 'ev_loss:', ev_loss , 'avg_loss:', avg_loss / 100, 'ev_value:', ev_metric)
    
    avg_loss = 0

    saved = False
    if min_loss == None: 
      min_loss = ev_metric
    elif ev_metric < min_loss:
      min_loss = ev_metric
      generator.save('generator.h5')
      decoder.save('decoder.h5')
      vision.save('vision.h5')
      print('Generator saved with ev_value:', ev_metric)
      saved = True
    
    date_time = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    postfix = ('saved_' if saved else '') + f"{np.mean(ev_metric):.6f}"
    out_img_path = f'{output_directory}/{date_time}_{postfix}.jpg'

    print_to_image(text_encoder_model, generator, test_words, out_img_path)



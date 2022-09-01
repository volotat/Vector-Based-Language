from genericpath import isfile
from tkinter import * 

from PIL import ImageTk,Image  

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import os


I_SZ = 128
R_SZ = 512

need_total_ans = 100
total_ans = 0
correct_ans = 0

#Reading words from the full dictionary file
with open('words.txt') as file:
  words = file.readlines()
  words = [line.rstrip() for line in words]

if os.path.isfile("history.csv"): 
  hist = pd.read_csv("history.csv")
else: 
  hist = pd.DataFrame()

if os.path.isfile("statistic.csv"): 
  stat = pd.read_csv("statistic.csv")
else: 
  stat = pd.DataFrame(columns=["Word","Accuracy","Correct","Total"])

ws = Tk()  
ws.title('Human Testing Interface')
ws.geometry('514x638')

welcome_screen = Frame(ws)
loading_screen = Frame(ws)
testing_screen = Frame(ws)

system_color = welcome_screen.cget("background")

def create_and_show_welcome_screen():
  welcome_screen.pack()

  Label(
    welcome_screen,
    text=f'Choose the size of the dictionary',
    font=("Arial", 18)
  ).grid(row=0, column=0, columnspan=2, sticky='nesw', pady=(140, 20))

  for ind, size in enumerate([10, 20, 40, 100, 200, 400, 1000, 2275]):
    if ind == 0:
      b = Button(welcome_screen, text='First ' + str(size), command = lambda s=size: _set_dict_size(s))
      b.grid(row=ind+1, column=0, columnspan=2, sticky='nesw', pady=(0,5))
    else:
      b = Button(welcome_screen, text='First ' + str(size), command = lambda s=size: _set_dict_size(s))
      b.grid(row=ind+1, column=0, sticky='nesw', pady=(0,5))

      b = Button(welcome_screen, text='Learn random 10', command = lambda s=size: _set_dict_size(s, crop_rnd = 10))
      b.grid(row=ind+1, column=1, sticky='nesw', pady=(0,5))

  def _set_dict_size(size, crop_rnd = -1):
    welcome_screen.pack_forget()
    create_and_show_loading_screen()

    ws.after(100, lambda s=size: set_dict_size(s, crop_rnd))

def set_dict_size(size, crop_rnd = -1):
  global word_dict, curr_word_ind, generator, words_as_imgs

  word_dict = words[:size]
  if crop_rnd > 0: 
    np.random.shuffle(word_dict)
    word_dict = word_dict[:crop_rnd]
  curr_word_ind = np.random.randint(len(word_dict))

  generator = tf.keras.models.load_model('generator.h5')

  print('Preparing images...')

  #load universal-sentence-encoder
  os.environ["TFHUB_CACHE_DIR"] = "tfhub-models/"
  text_encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  words_embeddings = text_encoder_model(word_dict).numpy()

  words_as_imgs = generator.predict(words_embeddings)

  loading_screen.pack_forget()
  create_and_show_testing_screen()


def create_and_show_loading_screen():
  loading_screen.pack()

  Label(
    loading_screen,
    text=f'Preparing images...',
    font=("Arial", 18)
  ).grid(row=0, column=0, sticky='nesw', pady=(140, 20))

def create_and_show_testing_screen():
  global canvas, image_container, tk_img, label, buttons_list
  testing_screen.pack()

  img = ((words_as_imgs[curr_word_ind] + 1) * 127.5).astype(np.uint8).reshape((I_SZ,I_SZ, 3))
  img = cv2.resize(img, (R_SZ,R_SZ), interpolation = cv2.INTER_AREA)
  canvas = Canvas(
    testing_screen, 
    width = R_SZ, 
    height = R_SZ
  )  
  canvas.grid(row=0, column=0, columnspan=2, pady=(0, 10))
  tk_img = ImageTk.PhotoImage(Image.fromarray(img))  
  image_container = canvas.create_image(0, 0, anchor=NW, image=tk_img) 


  label = Label(
    testing_screen,
    text=f'Guess what word this image represents',
    font=("Arial", 18)
  )
  label.grid(row=1, column=0, columnspan=2, pady=(0, 20))


  buttons_list = []
  curr_word = word_dict[curr_word_ind]
  options = list(filter(lambda w: w != curr_word, word_dict))
  np.random.shuffle(options)
  options = [curr_word] + options[:3]
  np.random.shuffle(options)
  for ind, word in enumerate(options):
    b = Button(testing_screen, text=word, command = lambda w=word: test_word(w))
    b.grid(row=2 + ind // 2, column=ind % 2, sticky='nesw')
    buttons_list.append(b)

def test_word(word):
  global total_ans, correct_ans, stat, hist

  word_data = stat[stat['Word'] == word_dict[curr_word_ind]]
  if len(word_data) > 0:
    w_correct_ans = int(word_data.iloc[0]['Correct'])
    w_total_ans = int(word_data.iloc[0]['Total'])
  else:
    w_correct_ans = 0
    w_total_ans = 0
    data = {"Word": word_dict[curr_word_ind],
          "Accuracy": '-',
          "Correct": w_correct_ans,
          "Total": w_total_ans}
    stat = stat.append(data, ignore_index=True)

  total_ans += 1
  w_total_ans += 1
  res_text = 'WRONG!'
  label.config(bg="salmon")
  if word == word_dict[curr_word_ind]:
    correct_ans += 1
    w_correct_ans += 1
    res_text = 'CORRECT!'
    label.config(bg="light green")

  stat.loc[stat['Word'] == word_dict[curr_word_ind], ['Accuracy','Correct', 'Total']] = f'{(w_correct_ans/w_total_ans):.4f}', w_correct_ans, w_total_ans
  print(f'{correct_ans}/{total_ans} | {correct_ans/total_ans * 100 :.2f}% | {res_text}')

  label.configure(text=f'Correct answer: {word_dict[curr_word_ind]}')
  [b.configure(state=DISABLED) for b in buttons_list]

  if total_ans >= need_total_ans:
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'Time: {dt_string} | Dict size: {len(word_dict)} | Accuracy: {(correct_ans/total_ans * 100):.2f}% | Correct: {correct_ans} | Total: {total_ans}')

    
    data = {"Time": dt_string,
            "DictSize": len(word_dict),
            "Accuracy": f'{(correct_ans/total_ans * 100):.2f}%',
            "Correct": correct_ans,
            "Total": total_ans}
    hist = hist.append(data, ignore_index=True)
    hist.to_csv("history.csv", encoding='utf-8', index=False)

    stat.to_csv("statistic.csv", encoding='utf-8', index=False)

    ws.after(2000, ws.quit)
  else:
    ws.after(2000, reset)
  

def reset():
  global curr_word_ind, tk_img, buttons_list

  #reset word index
  curr_word_ind = np.random.randint(len(word_dict))

  #reset image on canvas
  img = ((words_as_imgs[curr_word_ind] + 1) * 127.5).astype(np.uint8).reshape((I_SZ,I_SZ, 3))
  img = cv2.resize(img, (R_SZ,R_SZ), interpolation = cv2.INTER_AREA)
  tk_img = ImageTk.PhotoImage(Image.fromarray(img))  
  canvas.itemconfig(image_container, image=tk_img)

  #reset presented options
  curr_word = word_dict[curr_word_ind]
  options = list(filter(lambda w: w != curr_word, word_dict))
  np.random.shuffle(options)
  options = [curr_word] + options[:3]
  np.random.shuffle(options)

  for ind, word in enumerate(options):
    buttons_list[ind].configure(text=word, command = lambda w=word: test_word(w), state=NORMAL)

  #reset label 
  label.config(text=f'Guess what word this image represents')
  label.config(bg=system_color)

create_and_show_welcome_screen()
ws.mainloop()
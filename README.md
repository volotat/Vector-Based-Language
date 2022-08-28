# Vector-Based-Language

This repository contains source code for my article "Can humans speak the language of the machines?" that is available here: [to be published soon]


To generate new language use following command:  
```python3 train.py``` 

This will produce generator.h5 that could be used in visualization purposes. This script also downloads Universal Sentence Encoder into 'tfhub-models' folder and generates vision.h5 and decoder.h5 that required to continue the training process. You can also use pretrained models available in 'article' folder to produce the exact results presented in the article.  

To check the reconstruction rate use the following command:  
```python3 metric.py``` 

To generate image with visualization of one or more sentences use the following command:  
```python3 utils.py -t "sentence one" "sentence two" "and so on..." -o output.png -spl 3```

Finally to learn the language use the human-testing-interface that could be run with following command:  
```python3 testing_interface.py```

This interface also tracks your accuracy on the per session and per word bases. This could be obtained in the history.csv and statistic.csv files respectively.

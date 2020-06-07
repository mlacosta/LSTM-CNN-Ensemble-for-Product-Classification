# MeLi Data Challenge 2019 | Deep Learning

### Author: Mariano Leonel Acosta | Leaderboard #17 - 0.89764
https://ml-challenge.mercadolibre.com/final_results

I developed a predictive system for product classification with 1588 different categories. Using Natural Language Processing (NLP) combined with Deep Learning, I was able to analize over two millions product descriptions from [Mercado Libre](http:///www.mercadolibre.com.ar) and predict new cases with a balanced accuracy of 89,76%. 

The final model consists of a Neural Network ensemble, a combination of Long Short Term Memory RNNs (LSTM) and Convolutional Nets (CNN). Each sub-system was trained independently on differents subset of the dataset. Then, to make the final prediction, each output is combined using weighted sums.  

## Implementation
In order to try this project on your own, first you need to download the dataset (using Bash): 

```
$wget https://meli-data-challenge.s3.amazonaws.com/train.csv.gz 
$wget https://meli-data-challenge.s3.amazonaws.com/test.csv 
$wget https://meli-data-challenge.s3.amazonaws.com/sample_submission.csv 
$gunzip resources/train.csv.gz
```

(Alternately the resources can be downloaded manually from [HERE](https://ml-challenge.mercadolibre.com/downloads))

Next, simply run the *main.py* script. 

The following libraries are required:

* Numpy
* Pandas
* SciKit Learn
* Tensorflow
* Keras

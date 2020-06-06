# MeLi Data Challenge 2019 | Deep Learning

### Author: Mariano Leonel Acosta | Public Leaderboard #17 - 0.89764
https://ml-challenge.mercadolibre.com/final_results

I developed a predictive system for product classification with 1588 different categories. Using Natural Language Processing (NLP) combined with Deep Learning, I was able to analize over two millons product descriptions from [Mercado Libre](http:///wwww.mercadolibre.com.ar) and predict new ones with a balanced accuracy of 89,76%. 

The final model consist of an essemble with diverse kinds of Neural Networks, mostly Long Short Term Memory RNN (LSTM) and Convolutional Nets (CNN). Each sub-system was trainned indepently on differents subset of the dataset. Then, to make the final prediction, each output is combined by using weighted sums.  

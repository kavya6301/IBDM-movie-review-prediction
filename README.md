# IMDb-movie-review-prediction
## Problem Statement
To classify positive and negative reviews (Binary Sentiment Classification) using NLP models, given a dataset having 50k IMDb movie reviews \
A set of 25,000 highly polar movie reviews for training and 25,000 for testing \
**Data Reference:** https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 

## Approach
This kind of problems are known as sentiment analysis problems. We have reviews as some text, where we sometimes don't need all the words. We just need few patterns from some significantly important words to recognise its sentiment. So here, we do data preprocessing before prediction. \
We later tokenize data, applied TfidfVectorizer tools to transform the text data to feature vectors, for input to estimator. \
Then we used Scikit-learn NLP tools like **Logistic Regression** and **SGD Classifier** to perform on data & give better classification

## Observation
We can observe an accuracy of 75.12% for Regression, 58.29% for Linear SVM and 75.1% for Naive Bayes algorithm.

## Project Outline
Extract data into pandas from drive

```bash
 imdb_data=pd.read_csv('IMDB Dataset.csv')
 print(imdb_data.shape)
 imdb_data.head(10)
```


## Conclusion
We can see that both logistic regression and multinomial naive bayes model performing well compared to linear support vector machines. Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Textblob.

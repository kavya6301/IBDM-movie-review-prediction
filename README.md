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
![image](https://user-images.githubusercontent.com/65950195/158792151-0be7084e-984c-4a7e-ad27-31ea7c9b862d.png)

To make data easier for prediction, we do preprocessing like tokenizing the data, html parsers, stemming the words, removing regularised expressions and stop words, so on.

```bash
 #Removing the html strips
 def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

 #Removing the square brackets
 def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

 #Removing the noisy text
 def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
 #Apply function on review column
 imdb_data['review']=imdb_data['review'].apply(denoise_text)
```
Now, the normalised data will be observed as follows
```bash
 #normalized train reviews
 norm_train_reviews=imdb_data.review[:40000]
 norm_train_reviews[0]
 #convert dataframe to string
```
![image](https://user-images.githubusercontent.com/65950195/158796379-98a873b1-4895-41bd-af05-53a18b9c2ed2.png)

Vectorizing the data using tfidf vectorizer and count vectorizer for transforming the data into numbers and we use label binarizer for transforming sentiments to 0 or 1
```bash
 #Count vectorizer for bag of words
 cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
 #transformed train reviews
 cv_train_reviews=cv.fit_transform(norm_train_reviews)
 #transformed test reviews
 cv_test_reviews=cv.transform(norm_test_reviews)
```

```bash
 #labeling the sentiment data
 lb=LabelBinarizer()
 #transformed sentiment data
 sentiment_data=lb.fit_transform(imdb_data['sentiment'])
 print(sentiment_data.shape)
```

Now, our data is ready. We apply our models and get the accuracy score as follows for count vectorizer bag of words (bow) and tfidf vectorizer words:

-  lr_bow_score : 0.7512
-  lr_tfidf_score : 0.75
-  svm_bow_score : 0.5829
-  svm_tfidf_score : 0.5112
-  mnb_bow_score : 0.751
-  mnb_tfidf_score : 0.7509


## Conclusion
We can see that both logistic regression and multinomial naive bayes model performing well compared to linear support vector machines. Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Textblob.

Positive words in our data:/
![image](https://user-images.githubusercontent.com/65950195/158798603-b8351135-c05f-41ce-97ea-ead11ab5b91b.png)

Negative words in our data:/
![image](https://user-images.githubusercontent.com/65950195/158798640-83b6910b-fc64-453e-9b90-f558c06091b6.png)

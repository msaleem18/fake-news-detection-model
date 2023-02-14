## fake-news-detection-model


#### Overview
You’re a Machine Learning Engineer working for a new social media company called ABC Co. The data science team has uncovered a recent surge in the frequency with which news articles are shared over ABC’s platform. While this is good news for customer engagement, the product team is worried about the prevalence of misinformation on the platform. To help mitigate this issue, they have asked you to design a solution for identifying which articles are authentic and which are not.

#### Assumptions
There will be different types of fake news and this model is not equiped to handle all of them. In general we can expect fake news to fall in the following broad categories:

![alt text](https://github.com/msaleem18/fake-news-detection-model/blob/main/types_of_fake_news.png)

###### SOURCE: Open Science Index, Computer and Information Engineering Vol:14, No:12, 2020 waset.org/Publication/10011624

This model is only focused on the main text, it'll use the main text and the training labels to train the model and detect if its a fake news article.

Model Assumptions:
* Assuming all articles are in English 
* Assuming fake articles are based on the content as described above
* Assuming the dataset is balanced (this will not be true in real life scenarios)
* Articles with words less than 20 are mostly outliers

#### Methodology
1. READ DATA: from foler, please note the assumption is that the file is stored in your local drive. I have used current working directory function to grab the working directory and then read the files: pd.read_csv(current_wd+'/FILE_XYZ.csv')

2. DATA EXPLORATION: to understand data count, distribution and type of articles. 
* I have noticed that some (approx. 400) articles have less than 20 words in training set. Assuming they are outloers I dropped anything with less than 10 words
* 98% of the articles have less than equal to 1500 words, so max vocab is set at 1500 to limit vocabulary size
* Word exploration to add 'garbage words' as stop words and exclude them from model

3. PREPROCESSING: I have used spacy pipeline to preprocess text data. This includes:
* stop words removal + adding a few of my own stop words
* clean text such as remove special characters and remove https links
* tokenize, lemmatize and rejoin to create clean lemmatized sentences

4. FEATURE GENERATION: Genrate features such as word count, word count after clean up etc. After pre-processing and feature generation new file was stored in the same folder

5. SPLIT DATA: The train dataset was split with 90/10 ratio to create a new dataset for validation. The test dataset was not touched until the end

6. TF-IDF: TFIDF was used to generate TFIDF values for words based on train data only, later fitted on valiation and test data. This will be used to train ML Models

7. TRYING MODELS: Models tried for this project
* Naive Bayes
* Logistic Regression
* Random Forest
* LSTM
* LSTM + Random Forest
* BERT

8. HYPER-PARAMETERS: LSTM+Random Forest performed the best so that model combination was selected for final hyper parameter tuning. I used Hyperpot python library to find the best optimized parameters


#### Improvements
1. Better machines (with GPU) will allow to test more variations of LSTM and BERT model. With CPU only laptop training is extremely slow and can't be managed for larger epochs. Train Deep Learning models for larger epochs 
2. Use BERT embeddings to train a LSTM or other combination of model. The BERT embeddings have a larger vocab context and will help improve overall accuracy.
3. Get more data
4. Learn embeddings

## fake-news-detection-model


#### Overview:
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
1. Read data from foler, please note the assumption is that the file is stored in your local drive. I have used current working directory function to grab the working directory and then read the files.

2. Data exploration to understand data count, distribution and type of articles. 
* I have noticed that some (approx. 400) articles have less than 20 words in training set. Assuming they are outloers I dropped them
* 98% of the articles have less than equal to 1500 words, so max vocab is set at 1500 to limit vocabulary size
* Word exploration to add 'garbage words' as stop words and exclude them from model
* Removing low frequency words (less than 5)

3. Trying different models, starting with simpler models to set a base line and then moving up to more complex models.
* Linear Regression
* Random Forest
* LSTM
* LSTM + Random Forest

4. Results Summary:


#### Improvements


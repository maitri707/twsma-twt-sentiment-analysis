# COVID-19 Tweets Sentiment Analysis :bird:

## Overview
In this project, we aim to perform sentiment analysis on COVID-19 tweets using the BERT (Bidirectional Encoder Representations from Transformers) classifier model. 



## Data Preprocessing
The data will be preprocessed using the Preprocessor class, which performs various operations such as removing URLs, hashtags, mentions, and stop words. The class also performs lemmatization on the text.<br><br>
These preprocessing operation were performed on the data set:<br>
1) Deleting missing values
2) Deleting duplicate rows
3) Lowering texts
4) Removing mentions
5) Removing hashtags
6) Removing URLs
7) Removing punctuations
8) Removing html tags
9) Removing double spaces
10) Remving stop words
11) Removing numbers
12) Lemmatizing texts

## Data Analysis
In the data analysis part of the project, we started by examining the sentiment distribution in our dataset. We found that out of all tweets related to COVID-19, 19% were classified as *neutral*, 44% as *positive*, and 37% as *negative*.

* To get a sense of the most common words used in each category, we generated word clouds for the tweets in each sentiment category. The word clouds gave us a visual representation of the most frequent words used in each sentiment category, with larger words indicating higher frequency. Here are the word clouds for each category:
<p class="row" float="left" align="middle">
  <img src="/images/positive.png" width="250" height="200" title="Positive"/>
  <img src="/images/neutral.png" width="250" height="200" title="Neutral"/> 
  <img src="/images/negative.png" width="250" height="200" title="Negative"/>
</p>

* I also created a bar chart to display the most frequent words for each category. This can help us identify the common themes or topics associated with each category of tweets.

* I further explored the data to determine the most frequent origin countries of tweets. The most frequent origin countries of tweets in the analyzed dataset are: `Unknown`, `England`, `United States`, and `India`.

* I also analyzed the most frequent hashtags and mentions in the dataset. The most frequently used hashtags were `#coronavirus`, `#covid_19`, `#Coronavirus`, `#COVID2019`, and `#COVID19`. The most frequently mentioned accounts were `@realdonaldtrump`, `@youtube`, `@borisjohnson`, `@tesco`, and `@amazon`.

Overall, these analyses provided valuable insights into the sentiment and content of COVID-19 related tweets, as well as the countries and accounts most commonly associated with these tweets.


## Model
The `BertClassifier` class is used for building the model. It uses the **BERT** model as a base and adds fully connected layers on top for classification. The forward method of the class takes in the input text and its attention mask and returns the classification results.
```python
class BertClassifier(nn.Module):
    def __init__(self, class_num):
        super(BertClassifier, self).__init__()
        self.class_num = class_num
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, self.class_num),
        )

    def forward(self, input_ids, mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
        out = self.fc(pooled_output)

        return out
```
## Results 
Here are the results after **5** ephocs:<br>
Train accuracy: 97%<br>
Validation accuracy: 89%<br>
Test accuracy:  87%
<br>

## Confusion Matrix
The following figure is the confusion matrix of the test data set:
<p class="row" float="left" align="middle">
  <img src="/images/cfm.png" title="confusion matrix"/>
</p>






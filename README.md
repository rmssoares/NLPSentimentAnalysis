# NLP: Sentiment Analysis - Reviews

The present system analyses a *train dataset* of labelled (*Positive / Negative*) Rotten Tomatoes reviews through a *Naive Bayes* approach, and attempts to classify the sentiment of the reviews in three *test datasets*:

* The *dataset used for training*;
* A *test dataset of Rotten Tomatoes' reviews*;
* A *test dataset of Nokia reviews* (hence, a different domain).

For comparison, there is also a *Rule-based* approach. 

These two approaches **were already implemented** by *Dr. Chenghua Lin* for the course of *Data Mining & Visualization* in the *University of Aberdeen*. 
The authors analyzed these approaches thoroughly in a report format, and implemented an improved version of the *Rule-based* approach based in *sentence polarity*.

## Getting Started

The software provided was entirely done in *Java*. To start, clone the present repository into your local machine. If you're unaware of how to achieve this, please become familiar with the mechanisms of [GitHub](https://help.github.com/articles/set-up-git) repositories.

```
git clone git@github.com:thyriki/NLPSentimentAnalysis.git
```


### Prerequisites
Ensure that you have [Python 3.6](https://www.python.org/downloads/) installed and properly set up.

### Instructions
Navigate to the project's folder, and run the following command:

```
python SentimentPract2.py
```

## Authors

* **Ricardo Soares** - [thyriki](https://github.com/thyriki)
* **Juan García del Muro** - [astraey](https://github.com/astraey)

For any inquiries, feel free to open up an issue.

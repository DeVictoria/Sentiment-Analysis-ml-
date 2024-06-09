import re
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from utils import emoticons, contraction_mapping, pattern_Sentiment, pattern_topic



class SentimentClassifier:

    def __init__(self, train_data):
        self._train_data = train_data
        self._test_data = None
        self._vectorizer = None

    def get_sentiment_stats(self, test_data, quite=True):
        self._vectorizer = CountVectorizer()
        self._test_data = test_data
        refactor_metod = lambda text: self._refactor_text(text, pattern_Sentiment)
        y_pred_logistic, predict_y = self._logistic_model("Sentiment", refactor_metod)
        if not quite:
            print(classification_report(test_data["Sentiment"], y_pred_logistic))
        return y_pred_logistic, predict_y

    def get_sentiment_stats_use_date(self, test_data, quite=True):
        self._vectorizer = CountVectorizer()
        self._test_data = test_data
        refactor_metod = lambda text: self._refactor_text(text, pattern_Sentiment)
        y_pred_logistic, predict_y = self._logistic_model_with_time("Sentiment", refactor_metod)
        if not quite:
            print(classification_report(test_data["Sentiment"], y_pred_logistic))
        return y_pred_logistic, predict_y

    def get_topic_stats(self, test_data, quite=True):
        self._vectorizer = TfidfVectorizer()
        self._test_data = test_data
        refactor_metod = lambda text: self._refactor_text(text, pattern_topic)
        y_pred_logistic, predict_y = self._logistic_model("Topic", refactor_metod)
        if not quite:
            print(classification_report(test_data["Topic"], y_pred_logistic))
        return y_pred_logistic, predict_y

    def get_organization_stats(self, test_data, organization):
        test_data = test_data[test_data['Topic'] == organization]
        result = DataFrame(self.get_sentiment_stats(test_data)[0])

        value_counts = result.value_counts()
        total_count = len(result)
        percentages = (value_counts / total_count) * 100
        stats_organization = DataFrame({
            'Sentiment': list(value_counts.index),
            'Count': value_counts.to_list(),
            'Percentage': percentages.values
        })
        print(stats_organization)

    def change_to_5_sentiment(self, test_data):
        predict_y = self.get_sentiment_stats(test_data)[1]
        res_5_sentiment = []
        for mass in predict_y:
            irrelevant = mass[0]
            negative = mass[1]
            neutral = mass[2]
            positive = mass[3]
            most_likely = max(mass[0], mass[1], mass[2], mass[3])
            if most_likely == irrelevant:
                res_5_sentiment.append('irrelevant')
            elif most_likely == negative:
                if negative > 2 * (neutral + positive):
                    res_5_sentiment.append('negative')
                else:
                    res_5_sentiment.append('Sad')
            elif most_likely == neutral:
                res_5_sentiment.append('neutral')
            elif most_likely == positive:
                if positive > 2 * (neutral + negative):
                    res_5_sentiment.append('Excited')
                else:
                    res_5_sentiment.append('positive')
        return DataFrame(res_5_sentiment)

    def _logistic_model(self, name_y, refactor_metod):
        X_train = self._vectorizer.fit_transform(self._train_data["TweetText"].apply(refactor_metod))
        y_train = self._train_data[name_y]
        X_test = self._vectorizer.transform(self._test_data["TweetText"].apply(refactor_metod))

        model = LogisticRegressionCV(max_iter=10000)
        model.fit(X_train, y_train)

        return model.predict(X_test), model.predict_proba(X_test)

    def _logistic_model_with_time(self, name_y, refactor_metod):
        self._train_data['refactor_TweetText'] = self._train_data["TweetText"].apply(refactor_metod)
        y_train = self._train_data[name_y]
        self._test_data['refactor_TweetText'] = self._test_data["TweetText"].apply(refactor_metod)

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', CountVectorizer(), 'refactor_TweetText'),
                ('time', CountVectorizer(), 'TweetDate')
            ])
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegressionCV(max_iter=10000))
        ])
        model.fit(self._train_data, y_train)

        return model.predict(self._test_data), model.predict_proba(self._test_data)

    def _refactor_text(self, text: str, pattern):
        text = text.lower()
        for val in pattern.values():
            text = re.sub(val[0], val[1], text)

        for word in emoticons.keys():
            text = text.replace(word, emoticons[word])

        for word in contraction_mapping.keys():
            text = text.replace(word, contraction_mapping[word])

        text = re.sub(r"[\-\"`@#$%^&*(|)/~\[\]{\}:;+,._=']+", ' ', text)
        return text

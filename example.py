import pandas as pd

from SentimentClassifier import SentimentClassifier

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

classifier = SentimentClassifier(train_data)

# классификатор распределит новый входной твит на 3 класса: положительные, отрицательный и нейтральный.
print("-" * 50)
classifier.get_sentiment_stats(test_data, quite=False)
print("-" * 50)

# Способен исследовать временную информацию при определении настроения твита.
classifier.get_sentiment_stats_use_date(test_data, quite=False)
print("-" * 50)

# Возможность предсказать организацию (например, Apple) по Твиту.
classifier.get_topic_stats(test_data, quite=False)
print("-" * 50)

# Способность выполнять анализ настроений по отношению к заданной организации.
classifier.get_organization_stats(test_data, 'microsoft')
print("-" * 50)

# Расширение до более, чем 3 классов настроений (до 5-балльной шкалы).
new_5_sentiment_colum = classifier.change_to_5_sentiment(test_data)
print(new_5_sentiment_colum.value_counts())
print("-" * 50)

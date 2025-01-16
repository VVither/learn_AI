import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import numpy as np

# 1. Загрузка и предобработка данных
try:
    df = pd.read_csv('Практика 1/fake_news.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'fake_news.csv' не найден.")
    exit()

df = df.dropna()

print("Структура датасета:")
print(df.head())
print(df.info())


# Визуализация распределения классов
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Распределение классов')
plt.show()

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Векторизация текста
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 3. Обучение модели
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)
pac.fit(tfidf_train, y_train)

# 4. Оценка модели
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.4f}")
print("Отчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.title('Матрица ошибок')
plt.ylabel('Фактические значения')
plt.xlabel('Предсказанные значения')
plt.show()


# Визуализация наиболее часто встречающихся слов
def plot_word_cloud(text, title):
    words = ' '.join(text)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

plot_word_cloud(df[df['label'] == 'FAKE']['text'], "Облако слов для фейковых новостей")
plot_word_cloud(df[df['label'] == 'REAL']['text'], "Облако слов для реальных новостей")


# 5. ROC кривая и AUC
y_prob = pac.decision_function(tfidf_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='REAL')
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.show()


#6. Дополнительная проверка точности
# Проверяем точность на разных срезах данных
test_size_values = np.arange(0.1, 0.5, 0.1) # разные размеры тестовой выборки

for test_size in test_size_values:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность с test_size = {test_size:.1f}: {accuracy:.4f}")


# 7. Оптимизация модели, если точность менее 90%
if accuracy < 0.90:
    print("Точность ниже 90%, пробуем оптимизацию...")

    #Подбор параметров для TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df = 0.005, ngram_range=(1, 2))
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Подбор параметров для PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=2000, random_state=42, tol=1e-4, C=1.1)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность после оптимизации: {accuracy:.4f}")
    print("Отчет о классификации после оптимизации:")
    print(classification_report(y_test, y_pred))

    # Визуализируем матрицу ошибок после оптимизации
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title('Матрица ошибок после оптимизации')
    plt.ylabel('Фактические значения')
    plt.xlabel('Предсказанные значения')
    plt.show()

     # ROC кривая и AUC после оптимизации
    y_prob = pac.decision_function(tfidf_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='REAL')
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая после оптимизации')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("Точность модели >= 90%, оптимизация не требуется")
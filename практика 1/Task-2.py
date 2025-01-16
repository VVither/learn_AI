import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv("parkinsons.data")

# 1. Подготовка данных
# Удаляем столбец 'name', так как он не влияет на предсказание
data = data.drop('name', axis=1)

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('status', axis=1)
y = data['status']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Нормализация признаков
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Создание и обучение модели XGBoost
# Задаем параметры модели
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=1000,  # Увеличенное кол-во деревьев
    learning_rate=0.01,  # Уменьшенный learning rate
    max_depth=4,        # Максимальная глубина дерева
    subsample=0.8,      # Доля выборки для обучения каждого дерева
    colsample_bytree=0.8, # Доля признаков для обучения каждого дерева
    random_state=42,
)


# Обучение модели
model.fit(X_train_scaled, y_train)

# 4. Оценка модели

# Предсказания на тестовой выборке
y_pred = model.predict(X_test_scaled)
# Вычисление точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")

# Проверка достижения 95% точности
if accuracy >= 0.95:
    print("Точность 95% достигнута!")
else:
    print("Требуемая точность не достигнута.")
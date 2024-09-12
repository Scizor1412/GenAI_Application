from model import AI, DeepLearning
from dataset import Dataset

X_train = [
  [3, 1500, 'urban'],   # 3 bedrooms, 1500 sq ft, urban location
  [2, 850, 'suburban'],  # 2 bedrooms, 850 sq ft, suburban location
  [4, 2000, 'rural'],    # 4 bedrooms, 2000 sq ft, rural location
]

y_train = [
  350000,   # House price for the first input example
  180000,   # House price for the second input example
  400000    # House price for the third input example
]

X_pred = [
  [1, 500, 'urban'],   # 3 bedrooms, 1500 sq ft, urban location
  [3, 1200, 'suburban'],  # 2 bedrooms, 850 sq ft, suburban location
  [2, 980, 'rural'],    # 4 bedrooms, 2000 sq ft, rural location
]

if __name__ == '__main__':
    newModel = AI(algorithm='Random Forest', model_type='classification')
    newModel.fit(model='Transformers', X_train=X_train, y_train=y_train)
    newModel.predict(model='Transformers', X_pred=X_pred)

    X_train.reverse()
    X_pred.reverse()

    newDeepLearning = DeepLearning(algorithm='Support Vector Machines', model_type='prediction')
    newDeepLearning.fit(model='Transformers', X_train=X_train, y_train=[i + 100 for i in y_train], learning_rate = 0.001)
    newModel.predict(model='Reinforcement Learning', X_pred=X_pred)

    newDataset = Dataset(input_datas='./input_data.csv', labels=['ID', 'Value'])
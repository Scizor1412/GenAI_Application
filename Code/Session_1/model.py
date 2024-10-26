# 4. Xây dựng một class DeepLearning kế thừa từ AI. Trong class này có một hàm mở rộng là `train_on_epoch` (huấn luyện trên từng epoch) với đầu vào là `model, X_train, y_train, epoch` và hàm viết đè là `fit` có thêm tham số `learning_rate`.

class AI:
    def __init__(self, algorithm, model_type):
        self.algorithm = algorithm
        self.model_type = model_type

    def fit(self, model, X_train, y_train):
        print(model)
        print(X_train)
        print(y_train)

    def predict(self, model, X_pred):
        print(model)
        print(X_pred)

class DeepLearning(AI):
    def train_on_epoch(self, model, X_train, y_train, epoch):
        print(model)
        print(X_train)
        print(y_train)
        print(epoch)

    def fit(self, model, X_train, y_train, learning_rate):
        print(model)
        print(X_train)
        print(y_train)
        print(learning_rate)
#  `input_datas` và `labels`.

class Dataset:
    def __init__(self, input_datas, labels):
        self.input_datas = input_datas
        self.labels = labels
        print(input_datas)
        print(labels)
        
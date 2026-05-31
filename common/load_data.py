import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 读取数据
def get_data():
    # 1. 读取数据集
    data= pd.read_csv('../data/train.csv')

    # 2. 划分数据集和训练集
    X = data.drop(['label'], axis=1)
    y = data['label']
    x_train, x_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    # print(y_train)
    # print('------=========')
    # print(y_train.values)
    return x_train,x_test,y_train.values,y_test.values

if __name__ == '__main__':
    x_train,x_test,y_train,y_test = get_data()
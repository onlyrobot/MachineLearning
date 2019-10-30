'''
Author: 彭瑶
Date: 2019/10/7
Description: 朴素贝叶斯分类模型
'''


from collections import Counter
import math


def classify(model: dict, data: list) -> str:
    '''根据生成的模型对数据进行分类

    Args:
        model: 预先生成的先验概率和条件概率模型
        data: 需要进行分类的数据
    
    Returns:
        返回数据所属的类别
    '''
    def get_pro(label):
        '''获取参数指定类别下的概率'''
        label = label[1]
        sum = math.log(label[0])
        for d in data:
            sum += math.log(label[1][d]) if d in label[1] else 0
        return sum
    res = max(model.items(), key=get_pro)
    return res[0]


def init_model(data: list) -> dict:
    '''初始化模型

    利用给定数据生成先验概率和条件概率模型

    Args:
        data: 给定的数据

    Returns:
        返回生成好的模型
    '''
    label_counter = Counter([d[0] for d in data])
    total = len(data)
    model = {}
    for label in label_counter:
        sub_model = model[label] = (label_counter[label] / total, {})
        sub_data = [d for d in data if d[0] == label]
        sub_total = len(sub_data)
        data_counter = Counter(sub_data)
        for d in data_counter:
            sub_model[1][d[1]] = data_counter[d] / sub_total
    return model


def get_data() -> list:
    '''获取数据的函数，返回格式是键值对列表'''
    row_data = [['Sunny', 'Hot', 'High', 'Weak', 'False'], 
            ['Sunny', 'Hot', 'High', 'Strong', 'False'], 
            ['Overcast', 'Hot', 'High', 'Weak', 'True'],
            ['Rain', 'Mild', 'High', 'Weak', 'True'], 
            ['Rain', 'Cool', 'Normal', 'Weak', 'True'], 
            ['Rain', 'Cool', 'Normal', 'Strong', 'False'], 
            ['Overcast', 'Cool', 'Normal', 'Strong', 'True'], 
            ['Sunny', 'Mild', 'High', 'Weak', 'False'], 
            ['Sunny', 'Cool', 'Normal', 'Weak', 'True'], 
            ['Rain', 'Mild', 'Normal', 'Weak', 'True'], 
            ['Sunny', 'Mild', 'Normal', 'Strong', 'True'], 
            ['Overcast', 'Mild', 'High', 'Strong', 'True'], 
            ['Overcast', 'Hot', 'Normal', 'Weak', 'True'], 
            ['Rain', 'Mild', 'High', 'Strong', 'False']]
    data = []
    for d in row_data:
        for i in range(len(d) - 1):
            data.append((d[-1], d[i]))
    return data


def main():
    model = init_model(get_data())
    print(model)
    print('分类结果：', classify(model, ['Overcast', 'Mild', 'Normal', 'Weak']))


if __name__ == '__main__':
    main()
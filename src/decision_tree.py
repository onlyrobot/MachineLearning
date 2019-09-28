'''
Author: 彭瑶
Date: 2019/9/26
Description: 决策树(Decision Tree)ID3算法
'''


import math


class TreeNode:
    '''决策树的节点

    每一条从根节点到叶节点的路径表示一个决策，
    其中叶节点表示决策的结果(True or False)，
    其他节点表示属性以及属性的取值。

    这里使用了兄弟节点来实现一个节点拥有多个子节点。

    Attributes: 
        attr: 节点表示的属性
        child: 属性的取值和子节点组成的元组
        sibling: 兄弟节点，同样是属性的取值和子节点组成的元组
    '''

    def __init__(self, attr: str):
        self.attr = attr
        self.child = None
        self.sibling = None

    def add_child(self, attr_value: str, child: 'TreeNode'):
        '''添加子节点

        Args:
            attr_value: 属性的取值
            child: 子节点
        '''
        if self.child is None:
            self.child = (attr_value, child)
        else:
            sibling = self.child[1]
            while sibling.sibling is not None:
                sibling = sibling.sibling[1]
            sibling.sibling = (attr_value, child)

    def __str__(self):
        '''实现树形输出决策树'''
        res = []

        def preorder(root: TreeNode, space: int, attr_value: str):
            '''递归得到决策树的表示

            前序遍历的方法来递归得到决策树的表示，并存到变量res中

            Args:
                root: 决策树节点
                space: 控制缩进的空格符数
                attr_value: 属性取值
            '''
            res.append(space * ' ' + attr_value + ':' + root.attr + '\n')
            space += 5
            child = root.child
            while child is not None:
                preorder(child[1], space, child[0])
                child = child[1].sibling
            
        preorder(self, 0, '')
        return ''.join(res)


def generate_decision_tree(data: list, row_indexes: set=None, 
col_indexes: set=None) -> TreeNode:
    '''从给定数据中生成决策树

    Args:
        data: 调用get_data()方法获取的数据
        row_indexes: data中用于生成决策树的行，默认值表示所有行
        col_indexes: data中用于生成决策树的列，默认值表示所有列
    
    Returns:
        返回生成好的决策树
    '''
    if row_indexes is None:
        row_indexes = set(range(1, len(data)))
        col_indexes = set(range(0, len(data[0]) - 1))
    
    def get_entropy(attr_dict: dict) -> float:
        '''计算信息增益(Information Gain)后的熵

        在确定某个属性后，信息增益会使得熵值减小，函数的作用就是计算
        信息增益后的熵值

        Args:
            attr_dict: 计算信息增益后的熵时需要用到的属性的取值信息
        Returns:
            信息增益后的熵
        '''
        def get_sub_entropy(attr_value_dict: dict) -> float:
            '''计算属性在取定某个值后的熵'''
            part_1 = part_2 = 0
            for value in attr_value_dict.values():
                part_1 += value
                part_2 += value * math.log2(value)
            return (part_1, math.log2(part_1) - 1 / part_1 * part_2)

        total = part = 0
        for value in attr_dict.values():
            sub_entropy = get_sub_entropy(value)
            part += sub_entropy[0] * sub_entropy[1]
            total += sub_entropy[0]
        return part / total
    
    # 下面是计算每个属性的信息增益后的熵、获取最大信息增益的属性以及
    # 对属性每个取值递归生成子节点的详细过程。
    attr_dicts = {}
    for col_index in col_indexes:
        attr_value_dicts = attr_dicts[col_index] = {}
        for row_index in row_indexes:
            label = data[row_index][col_index]
            if label not in attr_value_dicts:
                attr_value_dicts[label] = {}
            value = data[row_index][-1]
            if value not in attr_value_dicts[label]:
                attr_value_dicts[label][value] = 0
            attr_value_dicts[label][value] += 1
    min_entropy_attr_dict = min(attr_dicts.items(), key = 
    lambda x: get_entropy(x[1]))
    root = TreeNode(data[0][min_entropy_attr_dict[0]])
    attr_value_dicts = min_entropy_attr_dict[1]
    for attr_value_dict in attr_value_dicts.items():
        if len(attr_value_dict[1]) == 1:
            root.add_child(attr_value_dict[0], TreeNode(
                list(attr_value_dict[1].keys())[0]))
            continue
        sub_row_indexes = [i for i in row_indexes if 
        data[i][min_entropy_attr_dict[0]] == attr_value_dict[0]]
        sub_col_indexes = col_indexes.copy()
        sub_col_indexes.remove(min_entropy_attr_dict[0])
        root.add_child(attr_value_dict[0], generate_decision_tree(
            data, sub_row_indexes, sub_col_indexes))
    return root


def get_data() -> list:
    '''获取训练的数据

    Returns:
        包含表头的表，第一行表示表头，其他每一行表示一种决策，
        最后一列表示决策的结果。
    '''
    return [['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'], 
            ['Sunny', 'Hot', 'High', 'Weak', 'False'], 
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


def main():
    '''生成并显示决策树'''
    decision_tree = generate_decision_tree(get_data())
    print(decision_tree)


if __name__ == '__main__':
    main()
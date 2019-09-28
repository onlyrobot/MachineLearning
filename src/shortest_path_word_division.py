'''
Author: 彭瑶
Date: 2019/9/28
Description: 用最短路径的方式来中文分词
'''


from queue import Queue


class GraphNode:
    '''图的节点

    Attributes:
        i: int，表示该节点对应于的词开始位置
        j: int，表示该节点对应于词的结束位置
        adjacency: list，节点的出边
        step: int，用于辅助记录最短路径
        pre_nodes: list，用于辅助记录最短路径，表示前驱节点
    '''
    def __init__(self, i: int, j: int):
        self.i, self.j = i, j
        self.adjacency = []
        self.step = -1
        self.pre_nodes = []


def shortest_path(start_node, end_node) -> list:
    '''得到从开始节点到结束节点的所有最短路径

    采用广度优先搜索方法，最后返回所有的最短路径，
    其中每条路径是一个节点列表

    Args: 
        start_node: 开始节点
        end_node: 结束节点

    Returns:
        得到的最短路径列表，其中每条最短路径是一个节点序列
    '''
    que = Queue()
    start_node.step = 0
    que.put(start_node)
    # 开始广度优先搜索最短路径，同时标记路径步数
    while not que.empty():
        node = que.get()
        if node == end_node:
            break
        for adj_node in node.adjacency:
            if adj_node.step == -1:
                adj_node.step = node.step + 1
                adj_node.pre_nodes.append(node)
                que.put(adj_node)
            elif adj_node.step == node.step + 1:
                adj_node.pre_nodes.append(node)
    else:
        return []
    # 获取路径
    def get_path(end_node):
        if end_node.step == 0:
            return [[end_node]]
        pathes = []
        for pre_node in end_node.pre_nodes:
            pre_pathes = get_path(pre_node)
            for pre_path in pre_pathes:
                pre_path.append(end_node)
                pathes.append(pre_path)
        return pathes
    return get_path(end_node)


def divide(words: set, sentence: str) -> str:
    '''实现分词

    Args:
        words: 词典
        sentence: 要被分词的句子

    Returns:
        用/分隔的分好的词
    '''
    n, m = len(sentence), len(max(words, key = lambda x: len(x)))
    nodes = set()
    # i_map用于查找所有开始位置为i的节点
    i_map = {}
    # 创建节点
    for i in range(n):
        i_map[i] = []
        for j in range(i + 1, min(i + m + 1, n + 1)):
            if sentence[i: j] in words:
                node = GraphNode(i, j)
                nodes.add(node)
                i_map[i].append(node)
    # 添加额外的开始、结束节点，方便寻找最短路径
    start_node = GraphNode(-1, 0)
    end_node = GraphNode(n, n + 1)
    i_map[n], i_map[n + 1] = [], []
    i_map[n].append(end_node)
    nodes.add(start_node)
    nodes.add(end_node)
    # 构建图
    for node in nodes:
        for adj_node in i_map[node.j]:
            node.adjacency.append(adj_node)
    # # 输出图
    # def print_graph(start_node, space_num = 0):
    #     if start_node.j == n + 1:
    #         return
    #     elif start_node.i != -1:
    #         print(space_num * ' ' + sentence[start_node.i: start_node.j])
    #         space_num += 3
    #     for adj_node in start_node.adjacency:
    #         print_graph(adj_node, space_num)
    # print('生成的图：')
    # print_graph(start_node)
    # 寻找最短路径，可能有多条
    pathes = shortest_path(start_node, end_node)
    return ['/'.join([sentence[i.i: i.j] for i in path]) for path in pathes]


def get_words() -> set:
    '''获取词典'''
    return { '他', '只', '只会', '诊断', '一', '一般', '的', '疾病',
            '病', '说', '说的', '的', '的确', '确实', '实在', '在', '在理' }


def main():
    print('他只会诊断一般的疾病：', divide(get_words(), '他只会诊断一般的疾病'))
    print('他说的确实在理：', divide(get_words(), '他说的确实在理'))


if __name__ == '__main__':
    main()

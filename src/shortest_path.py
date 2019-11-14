'''
Author: 彭瑶
Date: 2019/9/28
Description: 用最短路径的方式来中文分词
'''


from queue import Queue


class Node:
    '''图的节点

    Attributes:
        word: str，该节点表示的词
        adjacency: list，节点的出边
        step: int，用于辅助记录最短路径
        pre_nodes: list，用于辅助记录最短路径，表示前驱节点
    '''
    def __init__(self, word: str):
        self.word = word
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


def gen_graph(sentence: str, start_node: Node, 
end_node: Node, buf: dict, words: set, m: int) -> Node:
    '''生成分词的图
    
    递归的生成句子对应的分词路径组成的图

    Args:
        sentence: 要分词的句子
        start_node: 该句子前面的节点
        end_node: 终止节点
        buf: 用于记忆化搜索，降低复杂度
        words: 词典
        m: 单个词的最大长度
    '''
    if sentence not in buf:
        if not sentence:
            buf[sentence] = {end_node}
        else:
            nodes = buf[sentence] = set()
            for end in range(min(len(sentence), m), 0, -1):
                word = sentence[: end]
                if word in words:
                    node = Node(word)
                    gen_graph(sentence[end:], node, end_node, buf, words, m)
                    nodes.add(node)
            if not nodes:    # 未登录词处理
                node = Node(sentence[0])
                gen_graph(sentence[1:], node, end_node, buf, words, m)
                nodes.add(node)
    for node in buf[sentence]:
        start_node.adjacency.append(node)


def divide(words: set, sentences: list, all: bool=False) -> list:
    '''实现分词

    Args:
        words: 词典
        sentence: 要被分词的句子
        all: 指定是否返回所有可能的分词

    Returns:
        分好的词序列
    '''
    m = len(max(words, key = lambda x: len(x)))
    divided_sentences = []
    for sentence in sentences:
        start_node = Node('S')
        end_node = Node('D')
        gen_graph(sentence, start_node, end_node, {}, words, m)

        # 寻找最短路径，可能有多条
        pathes = shortest_path(start_node, end_node)
        if all:    # 返回所有的最短路径分词
            divided_sentences.append([[node.word for 
            node in path[1: -1]] for path in pathes])
        else:    # 返回其中一条最短路径
            divided_sentences.append([node.word 
            for node in pathes[0]][1: -1])
    return divided_sentences


def main():
    words = { '他', '只', '只会', '诊断', '一', '一般', '的', '疾病',
            '病', '说的', '他说', '的', '的确', '确实', '实在', '在', '在理' }
    print('他只会诊断一般的疾病：', divide(words, ['他只会诊断一般的疾病']))
    print('他说的确实在理：', divide(words, ['他说的确实在理'], True))


if __name__ == '__main__':
    main()

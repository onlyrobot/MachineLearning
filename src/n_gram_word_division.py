'''
Author: 彭瑶
Date: 2019/10/23
Description: n_gram中文分词
'''


import shortest_path_word_division as spwd
import math
from queue import PriorityQueue
import sys


class Node:
    def __init__(self, word):
        self.word = word
        self.adj = {}
        # self.step = -1
        # self.pre_nodes = set()

    def __lt__(self, other):
        return False
    
    def shortest_path(self):
        self.pre = None
        (pq := PriorityQueue()).put((self, 0))
        while not pq.empty():
            node, dis = pq.get()
            if hasattr(node, 'checked'):
                continue
            elif node.word == 'EOS':
                return node.get_path()[1: -1]
            node.checked = True
            for vertex, edge in node.adj.items():
                if (not hasattr(vertex, 'min_dis') 
                or dis + edge < vertex.min_dis):
                    vertex.pre, vertex.min_dis = node, dis + edge
                    pq.put((vertex, vertex.min_dis))
        return None

    def get_path(self):
        if self.pre is None:
            return [self.word]
        path = self.pre.get_path()
        path.append(self.word)
        return path

    # def __str__(self):
    #     if not self.adj:
    #         return self.word
    #     else:
    #         return self.word + list(self.adj.keys())[0].__str__()


class NGram:
    def __init__(self, train_data):
        self.model = {}
        self.words = {'BOS', 'EOS'}
        self.max_word_len = 0
        total, counts = 0, {}
        for data in train_data:
            last_word = 'BOS'
            for word in data:
                total += 1
                self.words.add(word)
                self.max_word_len = max(self.max_word_len, len(word))
                if (word, last_word) in counts:
                    counts[(word, last_word)] += 1
                else:
                    counts[(word, last_word)] = 1
                last_word = word
            if ('EOS', last_word) in counts:
                counts[('EOS', last_word)] += 1
            else:
                counts[('EOS', last_word)] = 1
        for count in counts:
            self.model[count] = counts[count] / total

    def get_pro(self, key):
        if key in self.model:
            return -math.log(self.model[key])
        else:
            return -math.log(1 / len(self.words))

    def gen_graph(self, sentence, last_word, buf):
        root = Node(last_word)
        if sentence not in buf:
            if not sentence:
                buf[sentence] = {Node('EOS')}
            else:
                nodes = buf[sentence] = set()
                for end in range(min(len(sentence), self.max_word_len), 0, -1):
                    if (word := sentence[: end]) in self.words:
                        nodes.add(self.gen_graph(sentence[end:], word, buf))
                if not nodes:    # 未登录词处理
                    nodes.add(self.gen_graph(sentence[1:], sentence[0], buf))
        for node in buf[sentence]:
            root.adj[node] = self.get_pro((node.word, last_word))
        return root

    def seg(self, sentences):
        results = []
        for sentence in sentences:
            graph = self.gen_graph(sentence, 'BOS', {})
            results.append(graph.shortest_path())
        return results
            
        # def helper(sentence, last_word):
        #     if sentence == '':
        #         return (self.get_pro(('EOS', last_word)), [])
        #     max_pro, max_division = 0., []
        #     for end in range(min(len(sentence), self.max_word_len), 0, -1):
        #         if (word := sentence[: end]) in self.words:
        #             sub_res = helper(sentence[end:], word)
        #             pro = self.get_pro((word, last_word)) + sub_res[0]
        #             if pro > max_pro:
        #                 max_pro = pro
        #                 max_division = sub_res[1]
        #                 max_division.append(word)
        #     if not max_division:    # 未登录词识别
        #         sub_res = helper(sentence[1:], sentence[0])
        #         max_pro = sub_res[0]
        #         max_division = sub_res[1]
        #         max_division.append(sentence[0])
        #     return (max_pro, max_division)
        
        # res = []
        # for sentence in sentences:
        #     res.append(helper(sentence, 'BOS')[1])
        #     res[-1].reverse()
        # return res


def main():
    n_gram = NGram([['我的', '世界'], ['我的', '父亲'], ['我', '的', '女人']])
    print(n_gram.model)
    print(n_gram.seg(['你是我的女人']))


if __name__ == '__main__':
    main()
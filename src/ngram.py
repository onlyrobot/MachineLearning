'''
Author: 彭瑶
Date: 2019/10/23
Description: N元语法中文分词和词性标注
'''


import math
from queue import PriorityQueue


class NGram:
    class Node:
        '''最短词路径的节点'''
        def __init__(self, word: str):
            self.word = word
            self.adj = {}

        def __lt__(self, other):
            return False
            
    def __init__(self, seg_sentences: list, pos_sentences: list=None, n: int=2):
        '''用数据进行模型初始化

        Args:
            seg_sentences: 切分好的句子
            pos_sentences: 标注好词性的句子
            n: N元语法中的N
        '''
        self.n = n
        # 初始化分词模型
        self.init_seg_model(seg_sentences)
        # 初始化词性标注模型
        self.init_pos_model(pos_sentences)

    def init_seg_model(self, seg_sentences: list):
        '''初始化词的切分模型'''
        prior_word = ('BOS',) * (self.n - 1)
        # 用word2word表示条件概率P(词|前面的词)
        self.word2word = {prior_word: {}}
        # 用prior_words表示所有词的先验概率
        self.prior_words = {prior_word: 0}
        # 用word表示所有出现过的词的概率
        self.words = {}
        # 下面开始构建隐马尔可夫概率模型
        for sentence in seg_sentences:
            prior_word = ('BOS',) * (self.n - 1)
            self.prior_words[prior_word] += 1
            for word in sentence:
                if word not in self.words:
                    self.words[word] = 1
                else:
                    self.words[word] += 1
                if word not in self.word2word[prior_word]:
                    self.word2word[prior_word][word] = 1
                else:
                    self.word2word[prior_word][word] += 1

                prior_word = (prior_word + (word,))[1:]
                if prior_word not in self.word2word:
                    self.word2word[prior_word] = {}
                    self.prior_words[prior_word] = 1
                else:
                    self.prior_words[prior_word] += 1
            else:
                if 'EOS' not in self.word2word[prior_word]:
                    self.word2word[prior_word]['EOS'] = 1
                else:
                    self.word2word[prior_word]['EOS'] += 1

        # 计算概率和一些计算一些参数，如总词数total、单个词最大长度max_len
        for prior_word in self.word2word:
            for word in self.word2word[prior_word]:
                # self.model[last_word][word] = ((self.model[last_word][word] + 1) / 
                # (self.counts[last_word] + len(self.words)))
                self.word2word[prior_word][word] /= self.prior_words[prior_word]
        self.max_len = len(max(self.words, key=lambda x: len(x)))
        self.total = sum(self.words.values())
        for word in self.words:
            self.words[word] /= self.total
        for prior_word in self.prior_words:
            self.prior_words[prior_word] /= self.total

    def init_pos_model(self, pos_sentences: list):
        '''初始化词性标注模型'''
        # 用pos2pos表示条件概率P(词性|上一个词性)
        self.pos2pos = {'BOS': {}, 'EOS': {}}
        # 用pos2word表示条件概率P(词|词性)
        self.pos2word = {}
        # 用poses表示所有词性的概率
        self.poses = {'BOS': 0}
        # 下面开始构建词性标注的隐马尔可夫模型
        for pos_sentence in pos_sentences:
            prior_pos = 'BOS'
            self.poses['BOS'] += 1
            for word, pos in pos_sentence:
                if pos not in self.pos2pos[prior_pos]:
                    self.pos2pos[prior_pos][pos] = 1
                else:
                    self.pos2pos[prior_pos][pos] += 1

                if pos not in self.pos2word:
                    self.pos2word[pos] = {word: 1}
                elif word not in self.pos2word[pos]:
                    self.pos2word[pos][word] = 1
                else:
                    self.pos2word[pos][word] += 1
                
                if pos not in self.pos2pos:
                    self.poses[pos] = 1
                    self.pos2pos[pos] = {}
                else:
                    self.poses[pos] += 1

                prior_pos = pos
            else:
                if 'EOS' not in self.pos2pos[prior_pos]:
                    self.pos2pos[prior_pos]['EOS'] = 1
                else:
                    self.pos2pos[prior_pos]['EOS'] += 1

        # 计算概率
        for pos in self.pos2pos:
            for sub_pos in self.pos2pos[pos]:
                self.pos2pos[pos][sub_pos] /= self.poses[pos]
        for pos in self.pos2word:
            for word in self.pos2word[pos]:
                self.pos2word[pos][word] /= self.poses[pos]

    def get_pos2word_prob(self, word: str, pos: str) -> float:
        '''获取条件概率P(词|词性)
        
        Args:
            word: 词
            pos: 词性
        Returns:
            条件概率P(词|词性)
        '''
        if pos in self.pos2word:
            if word in self.pos2word[pos]:
                return self.pos2word[pos][word]
            else:
                # return 1 / len(self.pos_model) ** 4
                return 0
        else:
            return 1 / len(self.pos2pos)

    def get_word2word_prob(self, word: str, prior_word: tuple) -> float:
        '''获取条件概率P(词|前面的词)

        获取条件概率，并针对一些未登录词进一步处理
        
        Args:
            word: 词
            prior_word: 前面的词
        
        Returns:
            条件概率P(词|前面的词)
        '''
        if prior_word in self.word2word:
            if word in self.word2word[prior_word]:
                return -math.log(self.word2word[prior_word][word])
            elif word in self.words:
                return (-math.log(self.words[word]) - 
                math.log(self.prior_words[prior_word]))
            else:
                return (-math.log(1 / self.total) - 
                math.log(self.prior_words[prior_word]))
        else:
            if word in self.words:
                return (-math.log(1 / self.total) - 
                math.log(self.words[word]))
            else:
                return -math.log(1 / self.total ** 2)

    def dijkstra(self, node: Node) -> list:
        '''用迪杰斯特拉算法求最短词路径

        输入为图的开始节点，要计算到结束节点（被标记为'EOS'）的路径
        
        Args:
            node: 开始节点
        
        Returns:
            开始节点到结束节点的最短词路径组成的列表
        '''
        def get_path(node: self.Node) -> list:
            '''用于最后得到最短路径

            用递归的方法来得到最短路径组成的列表
            
            Args:
                node: 结束节点

            Returns:
                最短路径组成的列表
            '''
            if node.pre is None:
                return [node.word]
            path = get_path(node.pre)
            path.append(node.word)
            return path

        node.pre = None
        pq = PriorityQueue()
        pq.put((node, 0))
        while not pq.empty():
            node, dis = pq.get()
            if hasattr(node, 'checked'):
                continue
            elif node.word == 'EOS':
                return get_path(node)[1: -1]
            node.checked = True
            for vertex, edge in node.adj.items():
                if (not hasattr(vertex, 'min_dis') 
                or dis + edge < vertex.min_dis):
                    vertex.pre = node
                    vertex.min_dis = dis + edge
                    pq.put((vertex, vertex.min_dis))
        return None

    def gen_graph(self, sentence: str, prior_word: tuple, buf: dict) -> 'Node':
        '''生成分词的带权图，从而能够使用迪杰斯特拉算法进行求解
        
        递归的生成句子对应的分词路径组成的图

        Args:
            sentence: 要分词的句子
            prior_word: 该句子前面的词
            buf: 用于记忆化搜索，降低复杂度

        Returns:
            图的开始节点
        '''
        root = self.Node(prior_word[-1])
        if sentence not in buf:
            if not sentence:
                buf[sentence] = {self.Node('EOS')}
            else:
                nodes = buf[sentence] = set()
                for end in range(min(len(sentence), self.max_len), 0, -1):
                    word = sentence[: end]
                    if word in self.words:
                        nodes.add(self.gen_graph(sentence[end:], 
                        (prior_word + (word,))[1:], buf))
                if not nodes and '' not in buf:    # 未登录词处理
                    for end in range(1, min(len(sentence), 2) + 1):
                        nodes.add(self.gen_graph(sentence[end:], 
                        (prior_word + (sentence[: end],))[1:], buf))
        for node in buf[sentence]:
            root.adj[node] = self.get_word2word_prob(node.word, prior_word)
        return root

    def viterbe(self, sentence: list) -> list:
        '''维特比算法，用于优化隐马尔可夫模型的求解
        
        Args:
            sentence: 已经分好词的待标注句子，如['我', '是', '句子']

        Returns:
            标注好的句子，如[('我', n), ('是', v), ('句子', n)]
        '''
        pathes, last_poses = [], {'BOS': 1.}
        for word in sentence:
            cur_poses = {}
            pathes.append({})
            for last_pos in last_poses:
                for cur_pos in self.pos2pos[last_pos]:
                    prob = last_poses[last_pos] * self.get_pos2word_prob(word, 
                    cur_pos) * self.pos2pos[last_pos][cur_pos]
                    if cur_pos not in cur_poses or prob > cur_poses[cur_pos]:
                        cur_poses[cur_pos] = prob
                        pathes[-1][cur_pos] = last_pos
            last_poses = cur_poses
        else:
            pos = max(last_poses, key=lambda x: last_poses[x])
            poses = [pos]
            for node in pathes[-1: 0: -1]:
                pos = node[pos]
                poses.append(pos)
            return [(word, pos) for word, pos in zip(sentence, poses[:: -1])]

    def seg(self, sentences: list) -> list:
        '''分词
        
        Args:
            sentences: 待切分的句子列表，如[['我是句子'], ['我也是句子']]]

        Returns:
            切好的句子列表，如[['我', '是', '句子'], ['我', '也', '是', '句子']]
        '''
        results = []
        for sentence in sentences:
            graph = self.gen_graph(sentence, ('BOS',) * (self.n - 1), {})
            results.append(self.dijkstra(graph))
        return results

    def pos(self, sentences: list) -> list:
        '''词性标注
        
        Args:
            sentences: 分好的词的句子列表，如
            [['我', '是', '句子'], ['我', '也', '是', '句子']]

        Returns:
            标注好的句子列表，如[[('我', n), ('是', v), ('句子', n)], 
            [('我', n), ('也', d), ('是', v), ('句子', n)]]
        '''
        result = []
        for sentence in sentences:
            result.append(self.viterbe(sentence))
        return result


def main():
    seg_sentences = [['我的', '世界'], ['我的', '父亲'], ['我', '的', '爱人']]
    pos_sentences = [[('我的', 'a'), ('完美', 'b'), ('世界', 'c')], 
    [('我的', 'a'), ('完美', 'd'), ('东西', 'c')]]
    gram3 = NGram(seg_sentences, pos_sentences, 3)
    print(gram3.seg(['你是我的爱人']))
    print(gram3.pos([['我的', '完美', '世界']]))


if __name__ == '__main__':
    main()
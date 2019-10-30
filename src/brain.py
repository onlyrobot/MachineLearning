'''
Author: 彭瑶
Date: 2019/10/10
Description: 实现大脑识别、抽象和组合的功能
'''


class Node:
    def __init__(self, nodes={}, links={}, symbol='', bias=0, part_of={}, abstract={}):
        self.nodes = nodes
        # TODO: link's symbol
        self.links = links
        self.symbol = symbol
        self.bias = bias
        self.part_of = part_of
        self.abstract = abstract

    def __str__(self):
        return '\n'.join([self.symbol.__str__(), self.nodes.__str__(), 
        self.links.__str__(), self.bias.__str__()])

    def match(self, node):

        return None


def main():
    near = Node(symbol='near')
    hello = Node(symbol='hello')
    hi = Node(symbol='hi')
    print(hi)
    # hello_hi = Node(nodes={0: hello, 1: hi}, symbol='hello/hi')
    # print(hello_hi)
    world = Node(symbol='world')
    hello_world = Node(nodes={0: near, 1: hello, 2: world}, links={(0, 1, 2): 0}, symbol='hello world')
    print(hello_world)
    return None


if __name__ == '__main__':
    main()
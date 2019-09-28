'''
Author: 彭瑶
Date: 2019/9/28
Description: 最大匹配法中文分词
'''


def fmm(words: set, sentence: str) -> str:
    '''前向最大匹配(Forward Maximum Match)

    Args:
        words: 词典
        sentence: 要分词的句子
    
    Returns:
        用/分隔的分好的词
    '''
    n = len(sentence)
    m = len(max(words, key = lambda x: len(x)))
    res = []
    i = 0
    while i < n:
        j = i + m if i + m < n else n
        while i < j:
            if sentence[i: j] in words:
                res.append(sentence[i: j])
                i = j
                break
            else:
                j -= 1
        else:
            res.append(sentence[i])
            i += 1
    return '/'.join(res)


def bmm(words: set, sentence: str) -> str:
    '''反向最大匹配(Back Maximum Match)
    
    Args:
        words: 词典
        sentence: 要分词的句子
    
    Returns:
        用/分隔好的词
    '''
    n = len(sentence)
    m = len(max(words, key = lambda x: len(x)))
    res = []
    j = n
    while j > 0:
        i = j - m if j - m > 0 else 0
        while i < j:
            if sentence[i: j] in words:
                res.append(sentence[i: j])
                j = i
                break
            else:
                i += 1
        else:
            res.append(sentence[i - 1])
            j -= 1
    res.reverse()
    return '/'.join(res)


def get_words() -> set:
    '''获取词典'''
    return { '他', '是', '研究生', '研究', '物化', '生物', '学', '国人',
            '化学', '的', '泰国', '泰国人', '人民', '很', '友好', '人' }


def main():
    '''匹配测试'''
    sentence = '他不是研究生物化学的'
    print(sentence, '的匹配结果：')
    print('正向最大匹配：', fmm(get_words(), sentence))
    print('反向最大匹配：', bmm(get_words(), sentence))

    sentence = '他不是研究生物化学的'
    print(sentence, '的匹配结果：')
    print('正向最大匹配：', fmm(get_words(), sentence))
    print('反向最大匹配：', bmm(get_words(), sentence))


if __name__ == '__main__':
    main()
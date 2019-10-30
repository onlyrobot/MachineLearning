'''
Author: 彭瑶
Date: 2019/9/28
Description: 最大匹配法中文分词
'''


def fmm(words: set, sentences: list) -> list:
    '''前向最大匹配(Forward Maximum Match)

    Args:
        words: 词典
        sentences: 要分词的句子
    
    Returns:
        分好的词序列
    '''
    m = len(max(words, key = lambda x: len(x)))
    divided_sentences = []
    for sentence in sentences:
        n = len(sentence)
        divided_sentence = []
        i = 0
        while i < n:
            j = i + m if i + m < n else n
            while i < j:
                if sentence[i: j] in words:
                    divided_sentence.append(sentence[i: j])
                    i = j
                    break
                else:
                    j -= 1
            else:
                divided_sentence.append(sentence[i])
                i += 1
        divided_sentences.append(divided_sentence)
    # return ['/'.join(ds) for ds in divided_sentences]
    return divided_sentences


def bmm(words: set, sentences: list) -> list:
    '''反向最大匹配(Back Maximum Match)
    
    Args:
        words: 词典
        sentences: 要分词的句子
    
    Returns:
        分好的词序列
    '''
    m = len(max(words, key = lambda x: len(x)))
    divided_sentences = []
    for sentence in sentences:
        n = len(sentence)
        divided_sentence = []
        j = n
        while j > 0:
            i = j - m if j - m > 0 else 0
            while i < j:
                if sentence[i: j] in words:
                    divided_sentence.append(sentence[i: j])
                    j = i
                    break
                else:
                    i += 1
            else:
                divided_sentence.append(sentence[i - 1])
                j -= 1
        divided_sentence.reverse()
        divided_sentences.append(divided_sentence)
    # return ['/'.join(ds) for ds in divided_sentences]
    return divided_sentences


def main():
    '''匹配测试'''
    words = { '他', '是', '研究生', '研究', '物化', '生物', '学', '国人',
            '化学', '的', '泰国', '泰国人', '人民', '很', '友好', '人' }
    sentences = ['他不是研究生物化学的']
    print(sentences, '的匹配结果：')
    print('正向最大匹配：', fmm(words, sentences))
    print('反向最大匹配：', bmm(words, sentences))

    sentences = ['泰国人民很友好']
    print(sentences, '的匹配结果：')
    print('正向最大匹配：', fmm(words, sentences))
    print('反向最大匹配：', bmm(words, sentences))


if __name__ == '__main__':
    main()
'''
Author: 彭瑶
Date: 2019/10/12
Description: 多种方式实现中文分词，并比较它们的效果
'''


import maximum_match as mm
import shortest_path as sp
import ngram


def get_words(path, encoding='utf-16'):
    word_file = open(path, 'r', encoding=encoding)
    words = set(word_file.read().split('  '))
    word_file.close()
    return words


def get_sentences(path, encoding='utf-16'):
    '''获取句子'''
    data_file = open(path, 'r', encoding=encoding)
    data = data_file.read().splitlines()
    data_file.close()
    return data


def get_seg_sentences(path, encoding='utf-16'):
    '''获取切分好的句子'''
    sentences = get_sentences(path, encoding=encoding)
    return [sentence.split('  ') for sentence in sentences]


def get_pos_sentences(path, encoding='utf-16'):
    '''获取标注好词性的句子'''
    sentences = get_sentences(path, encoding=encoding)
    pos_sentences = [sentence.split(' ')[1:] for sentence in sentences 
    if sentence != '']
    for sentence in pos_sentences:
        while '' in sentence:
            sentence.remove('')
        for i in range(len(sentence)):
            if sentence[i][0] == '[':
                sentence[i] = sentence[i][1:]
            else:
                j = sentence[i].find(']')
                if j != -1:
                    sentence[i] = sentence[i][:j]
            j = sentence[i].find('/')
            sentence[i] = (sentence[i][: j], sentence[i][j + 1:])
    return pos_sentences
                

def get_correct_ratio(answers, results):
    '''得到准确率'''
    correct_count, total_count = 0, 0
    for answer, result in zip(answers, results):
        total_count += len(result)
        pos, lookup_map = 0, set()
        for answer_word in answer:
            lookup_map.add((pos, pos + len(answer_word)))
            pos += len(answer_word)
        pos = 0
        for result_word in result:
            if (pos, pos + len(result_word)) in lookup_map:
                correct_count += 1
            pos += len(result_word)
    return correct_count / total_count


def get_recall_ratio(answers, results):
    '''得到召回率'''
    correct_count, total_count = 0, 0
    for answer, result in zip(answers, results):
        total_count += len(answer)
        pos, lookup_map = 0, set()
        for answer_word in answer:
            lookup_map.add((pos, pos + len(answer_word)))
            pos += len(answer_word)
        pos = 0
        for result_word in result:
            if (pos, pos + len(result_word)) in lookup_map:
                correct_count += 1
            pos += len(result_word)
    return correct_count / total_count


def get_f_measure(answers, results):
    '''得到F-measure'''
    correct_ratio = get_correct_ratio(answers, results)
    recall_ratio = get_recall_ratio(answers, results)
    return 2 * correct_ratio * recall_ratio / (correct_ratio + recall_ratio)


def get_pos_correct_ratio(answers, results):
    '''得到词性标注准确率'''
    total_count, correct_count = 0, 0
    for answer, result in zip(answers, results):
        for answer_word, result_word in zip(answer, result):
            total_count += 1
            if answer_word == result_word:
                correct_count += 1
    return correct_count / total_count


def print_ratio(answers, results):
    '''输出评价结果'''
    print('correct ratio:', end='\t')
    print(get_correct_ratio(answers, results), end='\t')
    print('recall ration:', end='\t')
    print(get_recall_ratio(answers, results), end='\t')
    print('F-measure:', end='\t')
    print(get_f_measure(answers, results))


def main():
    words = get_words('E:/DataSet/NLP/中文分词语料（山西大学提供）/' + 
    '训练语料（528250词，Unicode格式）.txt')
    seg_trian = get_seg_sentences('E:/DataSet/NLP/中文分词语料（山西大学提供）/' + 
    '训练语料（528250词，Unicode格式）.txt')
    seg_test = get_sentences('E:/DataSet/NLP/中文分词语料（山西大学提供）/' + 
    '测试语料(Unicode格式).txt')
    seg_test_answer = get_seg_sentences('E:/DataSet/NLP/中文分词语料（山西大学提供）/' + 
    '测试语料答案（Unicode格式）.txt')
    pos_train = get_pos_sentences('E:/DataSet/NLP/人民日报语料199801/' 
    + '199801.txt', 'ansi')
    pos_test = []
    
    # 词性标注里面的句子同时也是分词的训练集
    for data in pos_train:
        pos_test.append([dat[0] for dat in data])
        seg_trian.append([dat[0] for dat in data])
        for dat in data:
            words.add(dat[0])

    print('forward maximum match:')    # 正向最大匹配分词
    print_ratio(seg_test_answer, mm.fmm(words, seg_test))

    print('\nbackward maximum match:')    # 反向最大匹配分词
    print_ratio(seg_test_answer, mm.bmm(words, seg_test))

    print('\nshortest path segementation:')    # 最短路径分词
    print_ratio(seg_test_answer, sp.divide(words, seg_test))

    print('\n2-gram:')    # 2元文法分词以及词性标注
    gram2 = ngram.NGram(seg_trian, pos_train, n=2)
    print_ratio(seg_test_answer, gram2.seg(seg_test))
    print('pos correct ratio:\t', get_pos_correct_ratio(pos_train[:200], 
    gram2.pos(pos_test[:300])))

    print('\n3-gram:')    # 3元文法分词以及词性标注
    gram3 = ngram.NGram(seg_trian, pos_train, n=3)
    print_ratio(seg_test_answer, gram3.seg(seg_test))
    print('pos correct ratio:\t', get_pos_correct_ratio(pos_train[:200], 
    gram3.pos(pos_test[:300])))

    print('\n4-gram:')    # 4元文法分词以及词性标注
    gram4 = ngram.NGram(seg_trian, pos_train, n=4)
    print_ratio(seg_test_answer, gram4.seg(seg_test))
    print('pos correct ratio:\t', get_pos_correct_ratio(pos_train[:200], 
    gram4.pos(pos_test[:300])))

    # 几个测试用例
    print('几个测试用例：')
    print(gram3.seg(['大连港年吞吐量超七千万吨', '今天同事问了我一道面试题']))
    print(gram3.pos([['迈向', '充满', '希望', '的', '新', '世纪', '——', '一九九八年', '新年', 
    '讲话', '（', '附', '图片', '１', '张', '）'], ['希望', '是', '什么', '东西']]))


if __name__ == '__main__':
    main()
'''
Author: 彭瑶
Date: 2019/10/12
Description: 多种方式实现中文分词，并比较它们的效果
'''


import maximum_match_word_division as mm
import n_gram_word_division as ng


def get_words():
    # word_file = open('E:/DataSet/NLP/words/微软拼音.txt', 'r', encoding='ansi')
    # words = set(word_file.read().splitlines())

    # word_file.close()
    # # 添加额外的utf-8词库
    # file_names = ['搜狗万能词库.txt', '体育爱好者.txt', '标准大词库.txt']
    # for file_name in file_names:
    #     word_file = open('E:/DataSet/NLP/words/' + file_name, 'r', encoding='utf-8')
    #     words = words.union(set(word_file.read().splitlines()))
    #     word_file.close()
    # 添加训练语料库
    word_file = open('E:/DataSet/NLP/中文分词语料（山西大学提供）/测试语料答案' + 
    '（Unicode格式）.txt', 'r', encoding='utf-16')
    words = set(word_file.read().split('  '))
    word_file.close()
    return words


def get_sentences(path):
    data_file = open(path, 'r', encoding='utf-16')
    data = data_file.read().splitlines()
    data_file.close()
    return data


def get_correct_ratio(answers, results):
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
    correct_ratio = get_correct_ratio(answers, results)
    recall_ratio = get_recall_ratio(answers, results)
    return 2 * correct_ratio * recall_ratio / (correct_ratio + recall_ratio)


def print_ratio(answers, results):
    print('correct ratio:')
    print(get_correct_ratio(answers, results))
    print('recall ration:')
    print(get_recall_ratio(answers, results))
    print('F-measure:')
    print(get_f_measure(answers, results))


def main():
    words = get_words()
    sentences = get_sentences('E:/DataSet/word_division_sentence.txt')
    answer_sentences = get_sentences('E:/DataSet/word_division_answer.txt')
    answers = [sentence.split('  ') for sentence in answer_sentences]

    # print('forward maximum match:')
    # print_ratio(answers, mm.fmm(words, sentences))

    print('backward maximum match:')
    print_ratio(answers, mm.bmm(words, sentences))

    print('1-gram:')
    n_gram = ng.NGram(answers)
    print_ratio(answers, n_gram.seg(sentences))


if __name__ == '__main__':
    main()
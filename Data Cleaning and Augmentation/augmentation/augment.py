from eda import *
import random
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="原始数据的输入文件目录")
ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--num_aug", required=False, type=int, help="每条原始语句增强的语句数")
ap.add_argument("--alpha", required=False, type=float, help="每条语句中将会被改变的单词数占比")
args = ap.parse_args()

#输出文件
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#每条原始语句增强的语句数
num_aug = 8 #default
if args.num_aug:
    num_aug = args.num_aug

#每条语句中将会被改变的单词数占比
alpha = 0.1 #default
if args.alpha:
    alpha = args.alpha

def gen_eda(train_orig, output_file, num_aug=9):

    writer = open(output_file, 'w', encoding='UTF-8')
    lines = open(train_orig, 'r', encoding='UTF-8').readlines()

    print("正在使用EDA生成增强语句...")
    no = "0"
    i = 0
    j = 0
    seq = []

    for line in lines:
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        if no != parts[0]:
            for k in range(2):
                j += 1
                random.shuffle(seq)
                for _, sen in enumerate(seq):
                    writer.write(str(j) + "\t" + str(_) + "\t" + sen + '\n')
            seq.clear()
            i = 0
            j += 1
            no = parts[0]
        label = parts[3]
        sentence = parts[2]
        aug_sentences = eda(sentence, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            aug_sen = aug_sentence.split()
            new_aug_sen = "".join(aug_sen)
            seq.append(new_aug_sen + "\t" + label)
            writer.write(str(j) + "\t" + str(i) + "\t" + new_aug_sen + "\t" + label + '\n')
            i += 1
    '''
    for line in lines:
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        #label = parts[0]
        sentence = parts[1]
        aug_sentences = eda(sentence, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            aug_sen = aug_sentence.split()
            new_aug_sen="".join(aug_sen)
            writer.write(str(i) + "\t" + new_aug_sen + '\n')
            i += 1
    '''

    writer.close()
    print("已生成增强语句!")
    print(output_file)

if __name__ == "__main__":
    gen_eda(args.input, output, num_aug=num_aug)

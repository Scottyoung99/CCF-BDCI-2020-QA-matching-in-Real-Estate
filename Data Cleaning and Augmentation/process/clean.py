import random
import argparse
import os
import jionlp as jio

def clean(input_flie,output_file):
    writer = open(output_file, 'w',encoding='UTF-8')
    lines = open(input_flie, 'r',encoding='UTF-8').readlines()
    print("正在清洗...")
    for line in lines:
        parts = line[:-1].split(',')
        query=parts[0]
        reply=parts[1]
        label=parts[2]
        #sen = jio.remove_url(sen) # 删除文本中的 html 标签
        #sen = jio.remove_phone_number(sen, detail=True)  #删除电话号码
        #sen = jio.remove_exception_char(sen) #删除文本中的异常字符
        query = jio.clean_text(query)
        reply = jio.clean_text(reply)
        query = jio.tra2sim(query, mode='word')
        reply = jio.tra2sim(reply, mode='word')
        writer.write(query + "," + reply + "," + label + '\n')
    writer.close()
    print("清洗完成!")
    print(output_file)


def merge1(input_flie,output_file):
    writer = open(output_file, 'w', encoding='UTF-8')
    lines = open(input_flie, 'r', encoding='UTF-8').readlines()
    print("正在清洗连接...")
    for _,line in enumerate(lines):
        parts = line[:-1].split('\t')
        no = parts[0]
        sen0 = parts[1].split()
        # sen1 = jio.remove_stopwords(sen0, remove_time=True,remove_number=True,remove_non_chinese=True,re)
        sen1="".join(sen0)
        sen2 = jio.clean_text(sen1)
        writer.write(no + "\t" + sen2 + '\n')
    writer.close()
    print("清洗连接完成!")
    print(output_file)

def merge2(input_flie,output_file):
    #writer = open(output_file, 'w', encoding='UTF-8')
    lines = open(input_flie, 'r', encoding='UTF-8').readlines()
    print("正在清洗连接...")
    for _,line in enumerate(lines):
        parts = line[:-1].split(',')
        no = parts[0]
        sort = parts[1]
        label = parts[3]
        sen0 = parts[2].split()
        if not len(sen0):
            print("list  "+no+"  "+sort+"\n")
        # sen1 = jio.remove_stopwords(sen0, remove_time=True,remove_number=True,remove_non_chinese=True,re)
        sen1="".join(sen0)
        sen2 = jio.clean_text(sen1)
        if len(sen2)==0 :
            print("str  "+no+"  "+sort+"\n")
        #writer.write(no + "\t" + sort + "\t" + sen2 + "\t" + label + '\n')
    #writer.close()
    print("清洗连接完成!")
    print(output_file)

def linked(input_flie,output_file):
    writer = open(output_file, 'w', encoding='UTF-8')
    lines = open(input_flie, 'r', encoding='UTF-8').readlines()
    print("正在连接...")
    temp="0"
    i=0
    right=[]
    for line in lines:
        parts = line[:-1].split('\t')
        no = parts[0]
        label = parts[3]
        sen = parts[2]
        sen = jio.clean_text(sen)
        if temp != no:
            right=list(set(right))
            k=len(right)
            l=2
            while k>1:
                for _ in range(k-1):
                    new_sen="，".join(right[_:_+l])
                    writer.write(temp + "\t" + str(i) + "\t" + new_sen + "\t" + "1" + '\n')
                    i+=1
                k-=1
                l+=1
            '''
            if k > 2 :
                for _ in range(k-2):
                    new_sen="".join(right[_:_+3])
                    writer.write(temp + "\t" + i + "\t" + new_sen + "\t" + "1" + '\n')
                    i+=1
            if k > 3 :
                for _ in range(k-2):
                    new_sen="".join(right[_:_+4])
                    writer.write(temp + "\t" + i + "\t" + new_sen + "\t" + "1" + '\n')
                    i+=1
            '''
            right.clear()
            i=0
            temp=no
        if len(sen) == 0: continue
        if label=="1" :
            right.append(sen)
        writer.write(no + "\t" + str(i) + "\t" + sen + "\t" + label + '\n')
        i+=1
    writer.close()
    print("连接完成!")
    print(output_file)

if __name__ == "__main__":
    input="data/train.reply.tsv"
    output="data/train_reply_linked_final.tsv"
    linked(input, output)
    #merge2(input, output)
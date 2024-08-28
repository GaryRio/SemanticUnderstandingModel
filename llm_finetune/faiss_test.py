import torch
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification,TrainerCallback
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from peft import PeftModel
from torch.nn.parallel import DataParallel
import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


#1 读取数据
#data=pd.read_csv('/kaggle/input/updated-origin/updated_origin_data.csv')
#data.rename(columns={'客户问题': 'question','回答': 'answer'}, inplace=True)
data=pd.read_csv('/data/zhongyichen/wangwei/venv/Expanded_data_k10_g10.csv')
question=pd.DataFrame(columns=['question'])
question=data['question']
question.to_csv('question.csv')
print(question.head(),len(question),len(question))

#将question转为docs格式，用于向量嵌入和测试
question_data=CSVLoader('question.csv')
docs=question_data.load()
train_data, test_data = train_test_split(docs, test_size=0.2, random_state=42)
print(len(train_data),len(test_data))

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(train_data)

#2 嵌入(微调前)
# model_name="/data/zhongyichen/gr/llm-finetune/langchain/embedding/bge-large-zh-v1.5"
# embeddings=HuggingFaceEmbeddings(model_name=model_name)
# print(embeddings)

#2 微调后embedding
class CustomEmbeddings():
    #     embed: AutoModelForSequenceClassification=None
    #     tokenizer: AutoTokenizer=None
    def __init__(self, model, tokenizer):
        self.embed = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # 如果有多个GPU可用，则使用DataParallel封装模型
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.embed = DataParallel(self.embed)
        # self.embed = DDP(self.embed.to(self.device), device_ids=[self.device.index])
        self.embed.to(self.device)

    def embed_documents(self, texts):
        # 对输入文本进行编码
        inputs = self.tokenizer(texts,
                                max_length=128,
                                truncation=True,
                                padding=True,
                                return_tensors="pt")
        with torch.no_grad():
            # 获取模型的输出
            inputs = inputs.to(self.device)
            outputs = self.embed(**inputs)
            outputs=outputs.hidden_state[-1][:,0,:]#提取最后一层的cls

            allocated_memory = torch.cuda.memory_allocated(device=self.device)
            print(f"  Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
            #cls0=outputs.last_hidden_state[:,0,:]
            #cls0 = hidden_state[-1][:, 0, :] + hidden_state[-2][:, 0, :] + hidden_state[-3][:, 0, :] + hidden_state[-4][ :, 0, :]
        return outputs
# 加载基础模型
peft_model_id="./embedding_lora"
base_model = AutoModelForSequenceClassification.from_pretrained(peft_model_id, output_hidden_states=True)
# 加载LoRA模型
model = PeftModel.from_pretrained(base_model, peft_model_id)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
#embedding
embeddings = CustomEmbeddings(model, tokenizer)

def cache_test():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        allocated_memory = torch.cuda.memory_allocated(device=device)
        # cached_memory = torch.cuda.memory_cached(device=device)
        # max_allocated_memory = torch.cuda.max_memory_allocated(device=device)
        # max_cached_memory = torch.cuda.max_memory_cached(device=device)

        print(f"Device {i}:")
        print(f"  Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
        # print(f"  Cached memory: {cached_memory / (1024 ** 3):.2f} GB")
        # print(f"  Max allocated memory: {max_allocated_memory / (1024 ** 3):.2f} GB")
        # print(f"  Max cached memory: {max_cached_memory / (1024 ** 3):.2f} GB")


#3 FAISS向量存储初始化
print("向量化：")
cache_test()
faiss_index = FAISS.from_documents(split_docs, embeddings)
cache_test()
print('向量化完成！')

#4 检索测试1——给定一个问题，检索数据库中最相近的，通过答案是否相同判断检索是否正确
#4.1 单句测试
def test01(query,index):
    k=1
    docs_and_scores = index.similarity_search_with_score(query,k)
    #docs = faiss_index.similarity_search(query, k)
    for doc, score in docs_and_scores:
        row=doc.metadata['row']
        print(f"simi_question: {data.iloc[row]['question']},\nanswer:{data.iloc[row]['answer']} ,\n Score: {score}")

query = "连接失败？"
test01(query,faiss_index)


# 4.2 批量测试
def test02(test_data, data, index):
    right = 0
    error_row = []
    error_row2 = []
    n = len(test_data)
    for i in tqdm(range(0, n), desc='Test'):
        row = test_data[i].metadata['row']
        query = data.iloc[row]['question']
        answer = data.iloc[row]['answer']  # 原回答
        # print(row,query)
        # 数据库检索
        response = index.similarity_search_with_score(query, k=10)
        context = []
        for doc, score in response:
            row2 = doc.metadata['row']
            question = data.iloc[row2]['question']
            answer2 = data.iloc[row2]['answer']  # 检索到的相近问题的回答
            context.append({'row': row2, 'question': question, 'answer': answer2})

        # top1相似度判断
        #         if answer==answer2:
        #             right=right+1
        #         else:
        # #             error_row.append(row)
        # #             error_row2.append(row2)
        #             embedding1 = embeddings.embed_query(answer)
        #             embedding2 = embeddings.embed_query(answer2)
        #     #         embedding1 = embeddings.embed_query(query)
        #     #         embedding2 = embeddings.embed_query(question)
        #             # 计算相似度
        #             similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        #             if similarity>=0.75:
        #                 right+=1
        #             else:
        #                 error_row.append(row)
        #                 error_row2.append(row2)

        # top5命中判断
        for i in range(len(context)):
            if answer == context[i]['answer']:
                right += 1
                break
            if i == len(context) - 1:
                error_row.append(row)
    acc = right / len(test_data)
    print(f"acc={acc:.05f}")
    return error_row, error_row2

error_row,error_row2=test02(test_data,data,faiss_index)

#4.3 错误检索记录
error=pd.DataFrame(columns=['row','question','answer'])
error2=pd.DataFrame(columns=['row','question','answer'])
for i in range(len(error_row)):
    e0=data.iloc[error_row[i]]
    new_row = pd.DataFrame([{'row':error_row[i],'question':e0['question'],'answer':e0['answer']}])
    e2=data.iloc[error_row2[i]]
    new_row2 = pd.DataFrame([{'row':error_row2[i],'question':e2['question'],'answer':e2['answer']}])
    error=pd.concat([error,new_row],axis=0,ignore_index=True)
    error2=pd.concat([error2,new_row2],axis=0,ignore_index=True)

#保存为md文件
def save_markdown_to_file(markdown_data, filename):
    """将Markdown数据保存到文件中"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_data)

error.to_csv('error.csv')
error2.to_csv('error2.csv')
markdown_table = error.to_markdown(index=True)
save_markdown_to_file(markdown_table, 'error.md')
markdown_table = error2.to_markdown(index=True)
save_markdown_to_file(markdown_table, 'error2.md')

#5 模糊数据纠错
#检索结果作为答案
# def update01(data):
#     list1=[1499,1501,1503,1505,1506,1507,1508,1511,1514,1515,1522,1523,1525,1526,1528,1533,1535,1536,1537,1543,1544,1545,1550,2191,2439,2704,2722,2932,3955,4058,4278,4778,6151,6157,6174,6189,6195,6216,6241,6362,6365,6484,6495,6534,6590,6597]
#     list2=[4483,4483,4506,4494,4483,4474,4514,4514,4475,4465,4483,4453,4483,4506,4506,4465,4519,4474,4465,4519,4474,4460,4514,2190,6708,1552,2679,4097,1629,3201,3174,3049,6346,6453,6479,6307,6496,6521,6314,6093,6165,6373,6397,6414,6257,6586]
#     for i in range(len(list1)):
#         row=list1[i]
#         row2=list2[i]
#         data.loc[row,'answer']=data.loc[row2,'answer']
#     return data
# #原结果作为答案
# def update02(data):
#     list1=[239,503,1159,1336,1464,1905,2348,2678,2683,2690,2990,2992,2993,2995,3509,3518,3992,4455,4478,4518,4623,4738,6094,6286,6312,6416,6464,6500,6514,6546]
#     list2=[1461,475,4097,1348,3694,4002,3678,2695,2719,2675,2947,2947,2947,2951,4191,4312,2959,1524,1502,1529,4969,4723,6363,6105,6235,6120,6166,6166,6211,6235]
#     for i in range(len(list1)):
#         row=list1[i]
#         row2=list2[i]
#         data.loc[row2,'answer']=data.loc[row,'answer']
#     return data
# #赋值和修改
# def update03(data):
#     new1=data.loc[167,'answer']+data.loc[4321,'answer']
#     data.loc[4321,'answer']=new1
#     data.loc[167,'answer']=new1
#     new2=data.loc[3866,'answer']+data.loc[2187,'answer']
#     data.loc[3866,'answer']=new2
#     data.loc[2187,'answer']=new2
#     list3=[4097,2931,2933,2932,4096,4102]
#     for i in range(len(list3)):
#         row=list3[i]
#         data.loc[row,'answer']=data.loc[2947,'answer']
#     return data


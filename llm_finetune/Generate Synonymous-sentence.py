import numpy as np
import pandas as pd

#1 数据筛选
origin_data=pd.read_csv('./updated-origin/updated_origin_data.csv')

#保存为md文件
def save_markdown_to_file(markdown_data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_data)

# origin_data=origin_data.drop(index=1610)
# origin_data=origin_data.drop(index=6360)
# origin_data.loc[2764,'客户问题']='手机显示连接失败怎么办？'
# origin_data.loc[6609,'客户问题']='套餐可以转让吗'
# origin_data.loc[6665,'客户问题']='门店/站点/渠道商咨询加盟？'
# origin_data.loc[6776,'客户问题']='你们产品或服务的独特优势。'
# origin_data.loc[6776,'客户问题']='财务状况和稳定性？'
# origin_data.loc[6844,'客户问题']='你们研发团队的构成和实力如何？'
# origin_data.loc[6848,'客户问题']='你们这锂换电的技术优势'
# #保存csv
# origin_data.to_csv('cleaned_data.csv')
# #保存为markdown_table
# markdown_table = origin_data.to_markdown(index=True)
# save_markdown_to_file(markdown_table, 'cleaned_data.md')

# cleaned_data=pd.read_csv("cleaned_data.csv")
# len(cleaned_data)

# import re
# def clean_text(text):
#     text=text.lower()
#     text = re.sub(r'！。', '！', text)
#     text = re.sub(r'。！', '。', text)
#     return text
# origin_data=pd.read_csv('/kaggle/input/qa-origin/.csv')
# # origin_data.rename(columns={'客户问题': 'question','回答': 'answer'}, inplace=True)
# # origin_data['answer']=origin_data['answer'].apply(clean_text)
# # for i in range(413,417):
# #     origin_data.loc[i,'question']=origin_data.loc[i,'question']+',电柜？'
# # for i in range(1337,1348):
# #     origin_data.loc[i,'answer']=origin_data.loc[1328,'answer']
# # origin_data.loc[6159,'answer']=origin_data.loc[6391,'answer']

# # origin_data.to_csv('origin_data.csv')
# # markdown_table = origin_data.to_markdown(index=True)
# # save_markdown_to_file(markdown_table, 'origin_data.md')

# print(len(origin_data),origin_data.head())

#筛选答案出现次数少于k次的qa对，并选取1条作为生成原句
def filter_data(data,k):
    # 计算每个回答的出现次数
    answer_counts = data['answer'].value_counts()
    # 筛选出出现次数少于k次的回答
    rare_answers = answer_counts[answer_counts <= k].index
    # 根据筛选出的回答过滤原始 DataFrame
    filtered_df = data[data['answer'].isin(rare_answers)]
    #每种回答只取一种
    unique_answers_df = filtered_df.groupby('answer').head(1)
    return unique_answers_df

unique_answers_df=filter_data(origin_data,k=10)
print(len(unique_answers_df))
#保存答案到文件
unique_answers_df.to_csv('unique_qa_k10.csv',index=False)
markdown_table = unique_answers_df.to_markdown(index=True)
save_markdown_to_file(markdown_table, 'unique_qa_k10.md')

#2 同义句生成
def get_same(prompt, model="qwen-turbo"):
    client = OpenAI(
        #api_key="sk-",
        api_key='sk-',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    messages = [{"system":'你的任务是生成尽量简短的高质量同义句',
                 "role": "user",
                 "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

#3 批量生成
from io import StringIO
def get_same_question():
    #data=pd.read_csv('/kaggle/input/qa-text/unique_questions_answers.csv')
    data=pd.read_csv('unique_qa_k10.csv')
    df1=pd.DataFrame(columns=['question', 'answer'])
    for i in tqdm(range(0,len(data)),desc='generate'):
        prompt=f"""请结合《》中回答的内容，给出<>中问题的20个同义句，尽量使回复的风格和长度不同，禁止引用《》中的内容作为回复。
《{data.iloc[i]['answer']}》，<{'你们'+data.iloc[i]['question']+'?'}>"""
        #Prompt=f""" Please provide two synonymous sentences for the question in<>based on the content of the answer in<>. It is prohibited to quote the content in<>as a response.
#《{data.iloc[i]['回答']}》,<{'你们'+data.iloc[i]['客户问题']+'?'}>"""
        sames = get_same(prompt)
        #转成dataframe形式
        df=pd.DataFrame(columns=['question'])
        df = pd.read_csv(StringIO(sames))
        #df = pd.read_csv(StringIO(sames), header=None, names=['question', 'answer'], sep='\n')
        df2=pd.DataFrame(columns=['question', 'answer'])
        df2['question']=df
        df2['answer']=data.iloc[i]['answer']
        #print(df2)
        df1=pd.concat([df1, df2], axis=0, ignore_index=True)
    return df1

expand_data=get_same_question()

#保存扩展数据为csv文件
print(len(expand_data))
for i in range(len(expand_data)):
    expand_data.loc[i,'question']=expand_data.loc[i,'question'][3:]
expand_data.head()
expand_data.to_csv('expand_data_k10_g20.csv',index=False)

#将扩展数据与原数据合并
origin_data=pd.read_csv('/kaggle/input/updated-origin/updated_origin_data.csv')
#expand_data=pd.read_csv('/kaggle/input/expanddddddddddd/Expand_data_k5.csv')
data_sum=pd.concat([origin_data, expand_data])
print(len(data_sum))
data_sum.to_csv('Expanded_data_k10_g20.csv',index=False)
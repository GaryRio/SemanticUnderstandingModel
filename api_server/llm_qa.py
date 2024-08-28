import torch
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer, AutoModel, snapshot_download
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
from langchain.vectorstores import FAISS
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".csv"):
                file_list.append(os.path.join(filepath, filename))
    return file_list

def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'csv':
            loader = CSVLoader(file_path=one_file, csv_args={
                # 定界符：用于分隔字段的单字符字符串。它默认为','
                'delimiter': ',',
                # 引号字符：用于引用包含特殊字符的字段的单字符字符串，如定界符或者quotechar，或者包含换行符。它默认为'"'.
                'quotechar': '"',
                # 字段名称：如果在创建对象时没有作为参数传递，则在第一次访问或从文件中读取第一条记录时初始化该属性。
                'fieldnames': ['客户问题', '回答']
            })
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        # data = loader.load()
        # print(data)
        # docs.extend(data)
        docs.extend(loader.load())
    return docs

# 构造知识库
def buildknowledge(embeddings, persist_directory):
    # 目标文件夹
    tar_dir = [
        "/data/zhongyichen/gr/api_server/data",
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个doc的大小
        chunk_overlap=50  # 每个doc包含上一个结尾部分的大小
    )
    split_docs = text_splitter.split_documents(docs)

    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()

class ChatGLM4_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None

    def __init__(self, model_path: str, gen_kwargs: dict = None):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        print("完成本地模型的加载")

        if gen_kwargs is None:
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        self.gen_kwargs = gen_kwargs

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        # glm-4-9b-chat
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        model_inputs = model_inputs.to('cuda')
        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"

# model_dir = snapshot_download("ZhipuAI/glm-4-9b-chat")
# print("model_dir:", model_dir)
# model_dir: /home/zhongyichen/.cache/modelscope/hub/ZhipuAI/chatglm3-6b
# model_dir: /home/zhongyichen/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat
# model_dir: /home/zhongyichen/.cache/modelscope/hub/ZhipuAI/glm-4v-9b

# 大模型初始化
def init_llm():
    model_dir = '/home/zhongyichen/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat'
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    llm = ChatGLM4_LLM(model_path = model_dir, gen_kwargs=gen_kwargs)
    return llm

# knowledgeBase初始化
def qa_init_knowledge(llm, build=False):
    # 加载本地词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="/data/zhongyichen/gr/llm-finetune/langchain/embedding/bge-large-zh-v1.5")
    persist_directory = './chroma/large_sum'
    if build:
        buildknowledge(embeddings, persist_directory)
    # 构建完成后加载数据
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 我们所构造的 Prompt 模板
    template = """你是宇谷科技的智能客服，参考以下上下文来回答客户的问题。对于每个问题只返回其对应的回答，不要有多余的内容。如果你不知道答案，就说你不知道，不要试图编造答案。
    {context}
    客户问题: {question}
    回答:"""
    # 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

# 使用llm+knowledge回答
def qa_knowledge(question, qa_chain):
    result = qa_chain({"query": question})
    return result["result"]

# faiss初始化
def qa_init_faiss():
    embeddings = HuggingFaceEmbeddings(model_name="/data/zhongyichen/gr/llm-finetune/langchain/embedding/bge-large-zh-v1.5")
    # 构建langchain_faiss向量数据库
    # 1 数据处理
    data_path = '/data/zhongyichen/wangwei/venv/Expanded_data_k10_g10.csv'
    data = pd.read_csv(data_path)
    question = pd.DataFrame(columns=['question'])
    question = data['question']
    question.to_csv('question.csv')
    # 转为docs格式
    question_data = CSVLoader('question.csv')
    docs = question_data.load()
    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个doc的大小
        chunk_overlap=50  # 每个doc包含上一个结尾部分的大小
    )
    split_docs = text_splitter.split_documents(docs)
    # 2 FAISS向量存储初始化
    faiss_index = FAISS.from_documents(split_docs, embeddings)
    return data, faiss_index

# 使用llm+faiss回答
def qa_faiss(llm, faiss_index, data, query):
    context = []
    response = faiss_index.similarity_search_with_score(query, k=5)
    for doc, score in response:
        row2 = doc.metadata['row']
        question = data.iloc[row2]['question']
        answer = data.iloc[row2]['answer']  # 检索到的相近问题的回答
        context.append({'question': question, 'answer': answer})

    template = f"""使用<>中的上下文来回答《》中的问题，回答长度接近上下文中answer的平均长度。如果你不知道答案，就说你不知道，不要试图编造答案。”。
                     <{context}> 问题: 《{query}》有用的回答:"""
    result = llm(template)
    return result


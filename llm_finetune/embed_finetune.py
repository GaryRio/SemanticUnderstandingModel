import torch
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification,EvalPrediction,TrainerCallback
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.parallel import DataParallel

#1 读取数据
data=pd.read_csv('/data/zhongyichen/wangwei/venv/updated_origin_data.csv')

#1.1 构造queries（查询），corpus（语料库），以及relevant_docs（相关文档）数据集，用于embedding微调
#分离数据
queries = list(data['question'])
corpus = list(data['answer'])

# 构建语料库
# corpus是一个包含所有答案的列表；也可以使用字典形式存储，键为文档ID，值为文档内容
corpus_dict = {idx: ans for idx, ans in enumerate(corpus)}

# 构建查询与相关文档映射
# relevant_docs是一个列表，其中每个元素是一个字典，包含查询和相关文档ID
relevant_docs = [
    {'query': query, 'relevant_doc_id': idx}
    for idx, query in enumerate(queries)
]

train_data_fine = []
for doc in relevant_docs:
    train_data_fine.append({
        "text": queries[doc["relevant_doc_id"]] + "[SEP]" + corpus[doc["relevant_doc_id"]],
        "label": 1  # 相关文档标签
    })

# 划分
train_data_fine,valid_data_fine = train_test_split(train_data_fine,test_size=0.2,random_state=42)

#2 嵌入
model_name="/data/zhongyichen/gr/llm-finetune/langchain/embedding/bge-large-zh-v1.5"

#3 数据类
class QcDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        encoding = self.tokenizer(text,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_len,
                                  return_tensors='pt',
                                  add_special_tokens=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label,dtype=torch.float32)
        }

#4 模型、分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = QcDataset(train_data_fine,tokenizer)
valid_dataset = QcDataset(valid_data_fine,tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name,output_hidden_states=True)
#model = AutoModel.from_pretrained(model_name,output_hidden_states=True)
#model=HuggingFaceEmbeddings(model_name=model_name)
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
# 将模型移动到设备上
#model.to(device)

#5 微调
# 定义LoRA配置
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,  # 序列分类任务
#     inference_mode=False,
#     r=8,  # LoRA rank
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=['pooling']  # 指定目标模块
# )
# # 使用LoRA配置包装模型
# model = get_peft_model(model, peft_config)
# # 打印模型以查看LoRA层是如何插入的
# model.print_trainable_parameters()

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,#50步记录一次日志
    save_steps=100,#检查点步数
    gradient_checkpointing=True,#梯度检查点，只在需要时重新计算梯度
    evaluation_strategy="epoch",  # 每个epoch评估一次
    save_strategy="epoch",  # 每个epoch保存一次
    metric_for_best_model="eval_loss",  # 根据验证损失选择最佳模型
    greater_is_better=False,  # 验证损失越低越好
    report_to="none"  # 避免报告到任何跟踪服务
)
#早停策略
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int):
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.best_metric is None or metrics["eval_loss"] < self.best_metric:
            self.best_metric = metrics["eval_loss"]
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"Patience counter: {self.patience_counter}")

        if self.patience_counter >= self.early_stopping_patience:
            print("Early stopping triggered.")
            control.should_training_stop = True
callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]  # 早停耐心值设为3

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    callbacks=callbacks
)

# 开始训练
trainer.train()

peft_model_id="./embedding_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

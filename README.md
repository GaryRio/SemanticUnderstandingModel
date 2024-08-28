# 客服语义理解模型demo

# SemanticUnderstandingModel

**主要包含三部分：llm_finetune, api_server, vue_front_demo**

## llm_fintune

主要内容：微调llm和测试回答效果

Generate Synonymous_sentence.py：筛选数据（答案出现次数少于k次），通过qwen模型生成这些数据的同义句（具体生成通过prompt控制，生成错误率1/20左右）。

embed_finetune.py：通过原数据集构造queries（查询），corpus（语料库），以及relevant_docs（相关文档）数据集，对embedding模型微调。

faiss_test.py：基于数据集中的问题构建向量数据库，并测试检索能力——单句测试、批量测试，并将检索错误的数据保存成error文件。

llm_test：测试qa_chain（llm+knowledge）和RAG检索增强（llm+faiss)的生成情况。

## api_server

主要内容：本项目界面演示的服务器端，采用flask构建。提供两种问答模型的后端接口。

（1）**llm+knowledge**：使用langchain框架的chroma向量数据库构建知识库（向量数据库）、得到检索器，并持久化到硬盘，选择llm（glm-4-9b-chat），使用PromptTemplate构造prompt模版，通过RetrievalQA将llm、检索器、prompt模版构造成一个qa_chain。

优点：构造简便快捷、langchain框架扩展性好

缺点：过程不透明、不便评测

本项目知识库已完成构建，可以直接运行。

如需根据所需构建知识库，需要配置好data目录（知识库目录），同时将app.py中构建知识库部分的参数build改为True，即可进行知识库构建。

（2）**llm+faiss**：RAG检索增强方案，将faiss向量数据库检索到的k条最相近数据作为大模型上下文，将上下文和原输入一起作为输入传给大模型生成回复。

优点：其生成可信度可由向量数据库的检索能力保证（即由topk检索质量保证——扩展数据集1的top5命中率可达96.7%，top10命中率可达98.1%）

运行之前，需要先完成本地embedding模型和llm模型的配置，修改llm_qa.py中对应的目录位置。

启动流程：直接运行app.py

## vue_front_demo

主要内容：本项目界面演示的前端界面，采用vue完成开发。

启动前需要完成vue项目的环境配置。

需要先运行服务器端，得到api接口地址，在src/components/MainLLM.vue中修改请求的url。

启动流程：在本目录下运行命令*yarn serve*，即可启动前端

前端界面功能说明：提供两种模型可运行，llm+faiss和llm+knowledge。推荐使用llm+faiss效果更好。


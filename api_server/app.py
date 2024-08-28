from flask import Flask
from flask_cors import CORS
from flask import request
from llm_qa import *

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def hello_world():
#     return 'Hello World!'

def init_sys():
    # 初始化大模型
    global llm
    llm = init_llm()

    # 构建向量数据库
    global data, faiss_index
    data, faiss_index = qa_init_faiss()

    # 构建知识库
    global qa_chain
    qa_chain = qa_init_knowledge(llm, build=False)



@app.route('/llm_faiss', methods=["POST"])
def llm_faiss():
    question = request.get_json()['question']
    print(question)
    answer = qa_faiss(llm, faiss_index, data, question)
    # answer = "AI的回答.."
    return {
        'success': True,
        'result': answer
    }

@app.route('/llm_knowledge', methods=["POST"])
def llm_knowledge():
    question = request.get_json()['question']
    print(question)
    answer = qa_knowledge(question, qa_chain)
    # answer = "AI的回答.."
    return {
        'success': True,
        'result': answer
    }


if __name__ == '__main__':
    init_sys()
    app.run(host='0.0.0.0')

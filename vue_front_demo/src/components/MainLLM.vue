<template>
  <div class="main">
    <div class="history">
      <div style="height: 15px"></div>
      <div v-for="(item, i) in qa_history">
        <div class="show_user">
          <i class="el-icon-user" style="font-size: 20px;"></i> <span>{{ item['question'] }}</span>
        </div>

        <div class="show_ai">
          <i class="el-icon-chat-dot-square" style="font-size: 20px;"></i> <span>{{ item['answer'] }}</span>
        </div>
        <hr>
      </div>


    </div>
    <div class="ask">
      <el-row>
        <el-col :span="23"><div class="q_input">
          <el-input
              type="textarea"
              :autosize="{ minRows: 2, maxRows: 4}"
              placeholder="向我提问吧！(shift+enter换行，enter提交)"
              v-model="input_question"
              @keydown.native="Keydown">
          </el-input>
        </div></el-col>
        <el-col :span="1"><div class="q_submit">
          <el-button @click="submit()" type="info" icon="el-icon-position" style="font-size: 17px" circle></el-button>
        </div></el-col>
      </el-row>
      <div class="q_input">

      </div>

    </div>
  </div>
</template>

<script>
import { mapState } from 'vuex'
export default {
  name: "MainLLM",
  computed: {
    ...mapState(['llm'])
  },
  data() {
    return {
      input_question: '',
      user_question: '',
      qa_history: [
        // { question: 'question', answer: 'answer'},
      ]
    }
  },
  methods: {
    Keydown(e) {
      if (!e.shiftKey && e.keyCode == 13) {
        e.cancelBubble = true; //阻止冒泡行为
        e.preventDefault(); //取消事件的默认动作*换行
        this.submit()
      }
    },
    submit() {
      let question = this.input_question
      let answer = 'AI客服思考中...'
      this.input_question = ''
      if(question.trim() !== '') this.qa_history.push({ 'question': question, 'answer': answer})

      //获取ai回答
      let requestForm = {'question': question}
      this.$message({
        message: '已提交，请等待...',
      });
      // llm+faiss回答
      if(this.llm === 'faiss') {
        this.axios.post('http://192.9.200.17:5000/llm_faiss', requestForm).then((resp) => {
          let data = resp.data;
          // console.log(data);
          if (data.success) {
            this.$message({
              message: '成功！',
              type: 'success'
            });
            let ai_result = data.result
            this.qa_history[this.qa_history.length-1]['answer'] = ai_result
          } else {
            this.$message.error('失败');
          }
        })
      }
      // llm+knowledge回答
      else if(this.llm === 'knowledge') {
        this.axios.post('http://192.9.200.17:5000/llm_knowledge', requestForm).then((resp) => {
          let data = resp.data;
          // console.log(data);
          if (data.success) {
            this.$message({
              message: '成功！',
              type: 'success'
            });
            let ai_result = data.result
            this.qa_history[this.qa_history.length-1]['answer'] = ai_result
          } else {
            this.$message.error('失败');
          }
        })
      }


    }
  }
}
</script>

<style scoped>
.main {
  height: 580px;
}
.history {
  height: 500px;
  background-color: aliceblue;
  overflow: auto;
}

.ask {
  height: 50px;
  text-align: center;
}

.q_input {
  height: 50px;
  line-height: 50px;
}

.q_submit {
  height: 50px;
  line-height: 50px;
}

.show_user {
  font-size: 18px;
  text-align: left;
  margin-left: 15px;
}

.show_ai {
  font-size: 18px;
  text-align: left;
  margin-left: 15px;
}
</style>
<template>
  <div>
    <el-container>
      <el-header>
        <MainHeader></MainHeader>
      </el-header>

      <el-container>
        <el-container>
          <el-main>
            <el-tabs v-model='activeTab' :tab-position="tabPosition" @tab-click="handleClick">
              <el-tab-pane name='faiss' label="客服问答-llm+faiss">
                <MainLLM></MainLLM>
              </el-tab-pane>

              <el-tab-pane name='knowledge' label="客服问答-llm+knowledge">
                <MainLLM></MainLLM>
              </el-tab-pane>

              <el-tab-pane name='others' label="其他">
              </el-tab-pane>
            </el-tabs>
          </el-main>

          <el-footer>
            <MainFooter></MainFooter>
          </el-footer>
        </el-container>
      </el-container>
    </el-container>
  </div>

</template>

<script>
import MainHeader from "@/components/MainHeader";
import MainFooter from "@/components/MainFooter";
import MainLLM from "@/components/MainLLM";
import { mapMutations } from 'vuex'

export default {
  name: "Show",
  // computed: {
  //   ...mapState([''])
  // },
  data() {
    return {
      tabPosition: 'left',
      activeTab: 'faiss'
    }
  },
  methods: {
    ...mapMutations(['setllm']),
    handleClick(tab, event) {
      this.setllm(this.activeTab)
    }
  },
  components: {
    MainHeader,
    MainLLM,
    MainFooter
  }
}
</script>

<style>
.el-header, .el-footer {
  background-color: #B3C0D1;
  color: #333;
  text-align: center;
  line-height: 50px;
  height: 50px;
  padding: 0px;
}


.el-main {
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
  /*line-height: 580px;*/
  height: 620px;
}


body {
  margin: 0px;
  padding: 0px;
}
</style>
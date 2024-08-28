import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  strict: true,
  state: {
    llm: 'faiss',
  },
  getters: {
  },
  mutations: {
    setllm (state, llm) {
      state.llm = llm;
    },
  },
  actions: {
  },
  modules: {
  }
})

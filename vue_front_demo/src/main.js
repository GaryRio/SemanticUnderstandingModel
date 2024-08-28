import 'element-ui/lib/theme-chalk/index.css';

import axios from 'axios';
import ElementUI from 'element-ui';
import Vue from 'vue'
import VueAxios from 'vue-axios';

import App from './App.vue'
import router from './router'
import store from './store/index.js'


Vue.config.productionTip = false
Vue.use(ElementUI);
Vue.use(VueAxios, axios);

new Vue({router, store, render: h => h(App)})
    .$mount('#app')


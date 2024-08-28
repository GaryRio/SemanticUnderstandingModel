import Vue from 'vue'
import VueRouter from 'vue-router'

import Show from '../views/Show.vue'

Vue.use(VueRouter)

const routes =
    [
        {path: '/', redirect: '/show'},
        {
            path: '/show',
            name: 'Show',
            component: Show,
            meta: {
                title: 'AI客服问答'
            }
        },
        
    ]

const router = new VueRouter({
    mode: 'history',
    routes
})


export default router

(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-5483b230"],{"1da1":function(t,e,r){"use strict";r.d(e,"a",(function(){return a}));r("d3b7");function n(t,e,r,n,a,o,i){try{var s=t[o](i),c=s.value}catch(u){return void r(u)}s.done?e(c):Promise.resolve(c).then(n,a)}function a(t){return function(){var e=this,r=arguments;return new Promise((function(a,o){var i=t.apply(e,r);function s(t){n(i,a,o,s,c,"next",t)}function c(t){n(i,a,o,s,c,"throw",t)}s(void 0)}))}}},"52d9":function(t,e,r){"use strict";r.r(e);var n=function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("div",[r("p",{staticStyle:{"margin-top":"-10px",color:"#1890FF","text-decoration":"underline",cursor:"pointer"},on:{click:t.showDrawer}},[t._v(" Register now! ")]),r("a-drawer",{attrs:{title:"注册一个新账号",width:500,visible:t.visible,"body-style":{paddingBottom:"80px"}},on:{close:t.onClose}},[r("a-form",{attrs:{form:t.form,layout:"vertical","hide-required-mark":""}},[r("a-row",{attrs:{gutter:16}},[r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"用户名*"}},[r("a-input",{directives:[{name:"decorator",rawName:"v-decorator",value:["name",{rules:[{required:!0,message:"User name can not be empty"}]}],expression:"[\n                  'name',\n                  {\n                    rules: [{ required: true, message: 'User name can not be empty' }],\n                  },\n                ]"}],attrs:{placeholder:"Please enter user name"}})],1)],1)],1),r("a-row",{attrs:{gutter:16}},[r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"密码*"}},[r("a-input",{directives:[{name:"decorator",rawName:"v-decorator",value:["password",{rules:[{required:!0,message:"Password can not be empty"}]}],expression:"[\n\t\t\t\t\t\t\t  'password',\n\t\t\t\t\t\t\t  { rules: [{ required: true, message: 'Password can not be empty' }] },\n\t\t\t\t\t\t\t]"}],attrs:{type:"password",placeholder:"Password"}},[r("a-icon",{staticStyle:{color:"rgba(0,0,0,.25)"},attrs:{slot:"prefix",type:"lock"},slot:"prefix"})],1)],1)],1)],1),r("a-row",{attrs:{gutter:16}},[r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"性别"}},[r("a-select",{directives:[{name:"decorator",rawName:"v-decorator",value:["gender"],expression:"['gender']"}],attrs:{placeholder:"Please choose the gender"}},[r("a-select-option",{attrs:{value:"M"}},[t._v(" 男性 ")]),r("a-select-option",{attrs:{value:"F"}},[t._v(" 女性 ")])],1)],1)],1),r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"年龄段"}},[r("a-select",{directives:[{name:"decorator",rawName:"v-decorator",value:["age"],expression:"['age']"}],attrs:{placeholder:"Please choose the age"}},t._l(t.ageList,(function(e,n){return r("a-select-option",{attrs:{value:n}},[t._v(" "+t._s(e)+" ")])})),1)],1)],1)],1),r("a-row",{attrs:{gutter:16}},[r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"职业"}},[r("a-select",{directives:[{name:"decorator",rawName:"v-decorator",value:["occupation"],expression:"['occupation']"}],attrs:{placeholder:"Please choose the occupation"}},t._l(t.occupationList,(function(e,n){return r("a-select-option",{attrs:{value:n}},[t._v(" "+t._s(e)+" ")])})),1)],1)],1),r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"邮政编码"}},[r("a-input",{directives:[{name:"decorator",rawName:"v-decorator",value:["zipCode"],expression:"['zipCode']"}],attrs:{placeholder:"Please enter user zip-code"}})],1)],1)],1),r("a-row",{attrs:{gutter:16}},[r("a-col",{attrs:{span:12}},[r("a-form-item",{attrs:{label:"头像"}},[r("a-upload",{staticClass:"uploader",attrs:{name:"file","list-type":"picture-card","show-upload-list":!1,action:this.$store.state.requestPath+"uploadHeadPic","before-upload":t.beforeUpload},on:{change:t.handleChange}},[t.imageUrl?r("a-avatar",{attrs:{src:t.imageUrl,alt:"avatar",size:80}}):r("div",[r("a-icon",{attrs:{type:t.loading?"loading":"plus"}}),r("div",{staticClass:"ant-upload-text"},[t._v(" 上传 ")])],1)],1)],1)],1)],1)],1),r("div",{style:{position:"absolute",right:0,bottom:0,width:"100%",borderTop:"1px solid #e9e9e9",padding:"10px 16px",background:"#fff",textAlign:"right",zIndex:1}},[r("a-button",{style:{marginRight:"8px"},on:{click:t.onClose}},[t._v(" Cancel ")]),r("a-button",{attrs:{type:"primary"},on:{click:t.handleSubmit}},[t._v(" Submit ")])],1)],1)],1)},a=[],o=(r("b0c0"),r("96cf"),r("1da1"));function i(t,e){var r=new FileReader;r.addEventListener("load",(function(){return e(r.result)})),r.readAsDataURL(t)}var s={created:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t.getAllAge(),t.getAllOccupation();case 2:case"end":return e.stop()}}),e)})))()},data:function(){return{form:this.$form.createForm(this),visible:!1,ageList:null,occupationList:null,loading:!1,imageUrl:null,user:{name:"undefine",password:"undefine",gender:"undefine",age:null,occupation:null,zipCode:"undefine",headPic:"default.jpg"}}},methods:{getAllAge:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var r,n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$get(t.$store.state.requestPath+"allAge");case 2:r=e.sent,n=r.data,t.ageList=n.data;case 5:case"end":return e.stop()}}),e)})))()},getAllOccupation:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var r,n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$get(t.$store.state.requestPath+"allOccupation");case 2:r=e.sent,n=r.data,t.occupationList=n.data;case 5:case"end":return e.stop()}}),e)})))()},userRegister:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var r,n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.axios.post(t.$store.state.requestPath+"register/",t.user);case 2:r=e.sent,n=r.data,0==n.code?(t.onClose(),t.$message.success(" (｡･∀･)ﾉﾞ 注册成功~ ",1)):t.$message.error(" (￣△￣；) 出错啦~ ",2);case 5:case"end":return e.stop()}}),e)})))()},showDrawer:function(){this.visible=!0},onClose:function(){this.visible=!1,this.imageUrl=null},handleChange:function(t){var e=this;"uploading"!==t.file.status?"done"===t.file.status&&(i(t.file.originFileObj,(function(t){e.imageUrl=t,e.loading=!1})),this.user.headPic=t.file.response.data.src):this.loading=!0},beforeUpload:function(t){var e="image/jpeg"===t.type||"image/png"===t.type;e||this.$message.error("You can only upload JPG file!");var r=t.size/1024/1024<2;return r||this.$message.error("Image must smaller than 2MB!"),e&&r},handleSubmit:function(t){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function r(){return regeneratorRuntime.wrap((function(r){while(1)switch(r.prev=r.next){case 0:t.preventDefault(),e.form.validateFields((function(t,r){t||(e.user.name=r.name,e.user.password=r.password,e.user.gender=r.gender,e.user.age=r.age,e.user.occupation=r.occupation,e.user.zipCode=r.zipCode,e.userRegister())}));case 2:case"end":return r.stop()}}),r)})))()}}},c=s,u=(r("f1e3"),r("2877")),l=Object(u["a"])(c,n,a,!1,null,null,null);e["default"]=l.exports},"96cf":function(t,e,r){var n=function(t){"use strict";var e,r=Object.prototype,n=r.hasOwnProperty,a="function"===typeof Symbol?Symbol:{},o=a.iterator||"@@iterator",i=a.asyncIterator||"@@asyncIterator",s=a.toStringTag||"@@toStringTag";function c(t,e,r){return Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}),t[e]}try{c({},"")}catch(N){c=function(t,e,r){return t[e]=r}}function u(t,e,r,n){var a=e&&e.prototype instanceof v?e:v,o=Object.create(a.prototype),i=new R(n||[]);return o._invoke=P(t,r,i),o}function l(t,e,r){try{return{type:"normal",arg:t.call(e,r)}}catch(N){return{type:"throw",arg:N}}}t.wrap=u;var h="suspendedStart",f="suspendedYield",p="executing",d="completed",g={};function v(){}function m(){}function y(){}var w={};w[o]=function(){return this};var b=Object.getPrototypeOf,x=b&&b(b(C([])));x&&x!==r&&n.call(x,o)&&(w=x);var L=y.prototype=v.prototype=Object.create(w);function _(t){["next","throw","return"].forEach((function(e){c(t,e,(function(t){return this._invoke(e,t)}))}))}function E(t,e){function r(a,o,i,s){var c=l(t[a],t,o);if("throw"!==c.type){var u=c.arg,h=u.value;return h&&"object"===typeof h&&n.call(h,"__await")?e.resolve(h.__await).then((function(t){r("next",t,i,s)}),(function(t){r("throw",t,i,s)})):e.resolve(h).then((function(t){u.value=t,i(u)}),(function(t){return r("throw",t,i,s)}))}s(c.arg)}var a;function o(t,n){function o(){return new e((function(e,a){r(t,n,e,a)}))}return a=a?a.then(o,o):o()}this._invoke=o}function P(t,e,r){var n=h;return function(a,o){if(n===p)throw new Error("Generator is already running");if(n===d){if("throw"===a)throw o;return F()}r.method=a,r.arg=o;while(1){var i=r.delegate;if(i){var s=k(i,r);if(s){if(s===g)continue;return s}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if(n===h)throw n=d,r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);n=p;var c=l(t,e,r);if("normal"===c.type){if(n=r.done?d:f,c.arg===g)continue;return{value:c.arg,done:r.done}}"throw"===c.type&&(n=d,r.method="throw",r.arg=c.arg)}}}function k(t,r){var n=t.iterator[r.method];if(n===e){if(r.delegate=null,"throw"===r.method){if(t.iterator["return"]&&(r.method="return",r.arg=e,k(t,r),"throw"===r.method))return g;r.method="throw",r.arg=new TypeError("The iterator does not provide a 'throw' method")}return g}var a=l(n,t.iterator,r.arg);if("throw"===a.type)return r.method="throw",r.arg=a.arg,r.delegate=null,g;var o=a.arg;return o?o.done?(r[t.resultName]=o.value,r.next=t.nextLoc,"return"!==r.method&&(r.method="next",r.arg=e),r.delegate=null,g):o:(r.method="throw",r.arg=new TypeError("iterator result is not an object"),r.delegate=null,g)}function O(t){var e={tryLoc:t[0]};1 in t&&(e.catchLoc=t[1]),2 in t&&(e.finallyLoc=t[2],e.afterLoc=t[3]),this.tryEntries.push(e)}function j(t){var e=t.completion||{};e.type="normal",delete e.arg,t.completion=e}function R(t){this.tryEntries=[{tryLoc:"root"}],t.forEach(O,this),this.reset(!0)}function C(t){if(t){var r=t[o];if(r)return r.call(t);if("function"===typeof t.next)return t;if(!isNaN(t.length)){var a=-1,i=function r(){while(++a<t.length)if(n.call(t,a))return r.value=t[a],r.done=!1,r;return r.value=e,r.done=!0,r};return i.next=i}}return{next:F}}function F(){return{value:e,done:!0}}return m.prototype=L.constructor=y,y.constructor=m,m.displayName=c(y,s,"GeneratorFunction"),t.isGeneratorFunction=function(t){var e="function"===typeof t&&t.constructor;return!!e&&(e===m||"GeneratorFunction"===(e.displayName||e.name))},t.mark=function(t){return Object.setPrototypeOf?Object.setPrototypeOf(t,y):(t.__proto__=y,c(t,s,"GeneratorFunction")),t.prototype=Object.create(L),t},t.awrap=function(t){return{__await:t}},_(E.prototype),E.prototype[i]=function(){return this},t.AsyncIterator=E,t.async=function(e,r,n,a,o){void 0===o&&(o=Promise);var i=new E(u(e,r,n,a),o);return t.isGeneratorFunction(r)?i:i.next().then((function(t){return t.done?t.value:i.next()}))},_(L),c(L,s,"Generator"),L[o]=function(){return this},L.toString=function(){return"[object Generator]"},t.keys=function(t){var e=[];for(var r in t)e.push(r);return e.reverse(),function r(){while(e.length){var n=e.pop();if(n in t)return r.value=n,r.done=!1,r}return r.done=!0,r}},t.values=C,R.prototype={constructor:R,reset:function(t){if(this.prev=0,this.next=0,this.sent=this._sent=e,this.done=!1,this.delegate=null,this.method="next",this.arg=e,this.tryEntries.forEach(j),!t)for(var r in this)"t"===r.charAt(0)&&n.call(this,r)&&!isNaN(+r.slice(1))&&(this[r]=e)},stop:function(){this.done=!0;var t=this.tryEntries[0],e=t.completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(t){if(this.done)throw t;var r=this;function a(n,a){return s.type="throw",s.arg=t,r.next=n,a&&(r.method="next",r.arg=e),!!a}for(var o=this.tryEntries.length-1;o>=0;--o){var i=this.tryEntries[o],s=i.completion;if("root"===i.tryLoc)return a("end");if(i.tryLoc<=this.prev){var c=n.call(i,"catchLoc"),u=n.call(i,"finallyLoc");if(c&&u){if(this.prev<i.catchLoc)return a(i.catchLoc,!0);if(this.prev<i.finallyLoc)return a(i.finallyLoc)}else if(c){if(this.prev<i.catchLoc)return a(i.catchLoc,!0)}else{if(!u)throw new Error("try statement without catch or finally");if(this.prev<i.finallyLoc)return a(i.finallyLoc)}}}},abrupt:function(t,e){for(var r=this.tryEntries.length-1;r>=0;--r){var a=this.tryEntries[r];if(a.tryLoc<=this.prev&&n.call(a,"finallyLoc")&&this.prev<a.finallyLoc){var o=a;break}}o&&("break"===t||"continue"===t)&&o.tryLoc<=e&&e<=o.finallyLoc&&(o=null);var i=o?o.completion:{};return i.type=t,i.arg=e,o?(this.method="next",this.next=o.finallyLoc,g):this.complete(i)},complete:function(t,e){if("throw"===t.type)throw t.arg;return"break"===t.type||"continue"===t.type?this.next=t.arg:"return"===t.type?(this.rval=this.arg=t.arg,this.method="return",this.next="end"):"normal"===t.type&&e&&(this.next=e),g},finish:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.finallyLoc===t)return this.complete(r.completion,r.afterLoc),j(r),g}},catch:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.tryLoc===t){var n=r.completion;if("throw"===n.type){var a=n.arg;j(r)}return a}}throw new Error("illegal catch attempt")},delegateYield:function(t,r,n){return this.delegate={iterator:C(t),resultName:r,nextLoc:n},"next"===this.method&&(this.arg=e),g}},t}(t.exports);try{regeneratorRuntime=n}catch(a){Function("r","regeneratorRuntime = r")(n)}},b0c0:function(t,e,r){var n=r("83ab"),a=r("9bf2").f,o=Function.prototype,i=o.toString,s=/^\s*function ([^ (]*)/,c="name";n&&!(c in o)&&a(o,c,{configurable:!0,get:function(){try{return i.call(this).match(s)[1]}catch(t){return""}}})},d08c:function(t,e,r){},f1e3:function(t,e,r){"use strict";r("d08c")}}]);
//# sourceMappingURL=chunk-5483b230.4e6a93ec.js.map
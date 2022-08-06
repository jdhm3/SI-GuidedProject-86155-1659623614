(self.__LOADABLE_LOADED_CHUNKS__=self.__LOADABLE_LOADED_CHUNKS__||[]).push([[7740],{7111:(e,t,n)=>{n.d(t,{Z:()=>r});var o=n(898781),i=n(883119),s=n(785893);const r=({isInModal:e})=>{const t=(0,o.ZP)();return(0,s.jsx)(i.xu,{alignItems:"center",display:"flex",height:"100%",justifyContent:"center",left:!0,position:e?"absolute":"fixed",top:!0,width:"100%",zIndex:new i.Ry(1),dangerouslySetInlineStyle:{__style:{backgroundColor:"rgba(255, 255, 255, .5)"}},children:(0,s.jsx)(i.$j,{accessibilityLabel:t._('Loading', 'Full page loading state', 'Full page loading state'),show:!0})})}},638543:(e,t,n)=>{n.d(t,{Z:()=>w});var o=n(667294),i=n(966338),s=n(711781),r=n(472644),a=n(721809),l=n(898781),c=n(925927),d=n(883119),u=n(785893);function h(){var e;const t=(0,l.ZP)(),[n,i]=(0,o.useState)();return(null===(e=(0,c.k6)().location.state)||void 0===e?void 0:e.isAppFactoryOAuth)?null:(0,u.jsxs)(d.xu,{position:"relative",children:[(0,u.jsx)(d.iP,{onTap:()=>i(!0),children:(0,u.jsxs)(d.xu,{position:"relative",display:"flex",alignItems:"center",justifyContent:"center",height:40,rounding:"pill",color:"secondary",children:[(0,u.jsx)(d.xu,{height:24,width:24,marginTop:1,children:(0,u.jsx)(r.Z,{size:20})}),(0,u.jsx)(d.xu,{marginStart:2,children:(0,u.jsx)(d.xv,{weight:"bold",children:t._('Connect with Google', 'googleConnectButtonWithDisabledSDK.buttonText', 'Connect with Google button text')})})]})}),n&&(0,u.jsx)(a.Z,{type:"google",onDismiss:()=>i(!1)})]})}var g=n(304930),_=n(862624),p=n(524172),b=n(83969),f=n(867820);class m extends o.Component{constructor(...e){var t,n,i;super(...e),t=this,n="gsiButtonRef",i=(0,o.createRef)(),n in t?Object.defineProperty(t,n,{value:i,enumerable:!0,configurable:!0,writable:!0}):t[n]=i}componentDidMount(){const{isAuthenticated:e,isGoogleOneTapDisabled:t,isGoogleOneTapInitialized:n}=this.props;t||e||!n?(this.initializeAndRenderGSIButton(),e||n||(0,f.My)("mweb.gsi_button_fix.load_and_initialize_gsi_script")):(0,b.ru)({buttonRef:this.gsiButtonRef,isDesktop:!1})}componentDidUpdate(){const{isGoogleOneTapDisabled:e,isGoogleOneTapInitialized:t}=this.props;t?(0,b.ru)({buttonRef:this.gsiButtonRef,isDesktop:!1}):e&&this.initializeAndRenderGSIButton()}async initializeAndRenderGSIButton(){const{enabledITPBroswerFlow:e,onGoogleConnectSuccess:t}=this.props;await(0,b.Px)({unsafe:!0}),(0,b.OY)({itpSupport:e},(e=>{const n=e.credential;(0,f.My)(`unauth_mweb.gsi_button.select_by.${e.select_by}`),(0,f.NC)("press_google_connect_button");const o=(0,s.Z)(n);t({google_open_id_token:n,autologin:!1},{google_open_id_token:n,first_name:o.given_name,last_name:o.family_name,image_url:o.picture})})),(0,b.ru)({buttonRef:this.gsiButtonRef,isDesktop:!1})}shouldComponentUpdate(e){const{isGoogleOneTapDisabled:t,isGoogleOneTapInitialized:n}=this.props;return!n&&!!e.isGoogleOneTapInitialized||!t&&!!e.isGoogleOneTapDisabled}render(){return(0,u.jsx)(d.xu,{position:"relative",children:(0,u.jsx)(d.iP,{onTap:()=>(0,f.My)("gsi_personalized_button.tap"),children:(0,u.jsx)(d.xu,{height:44,children:(0,u.jsx)(d.xu,{ref:this.gsiButtonRef,position:"absolute",width:"100%"})})})})}}function x(e){const{isGoogleOneTapInitialized:t,isGoogleOneTapDisabled:n,browserType:o,platform:s}=(0,i.useSelector)((({googleOneTap:e,session:t})=>({isGoogleOneTapInitialized:e.googleOneTapInitialized,isGoogleOneTapDisabled:e.googleOneTapDisabled,browserType:t.browserType,platform:t.userAgentPlatform})));return(0,u.jsx)(m,{...e,enabledITPBroswerFlow:"ios"===s&&(2===o||1===o),isGoogleOneTapInitialized:t,isGoogleOneTapDisabled:n})}function y({onGoogleConnectSuccess:e}){const t=(0,o.useRef)(null),n=(0,p.M)(),i=(0,o.useRef)(e);return(0,o.useEffect)((()=>{i.current=e}),[e]),(0,o.useEffect)((()=>{const e=({credential:e,select_by:t})=>{(0,f.My)(`unauth_mweb.gsi_button.select_by.${t}`),(0,f.NC)("press_google_connect_button");const n=(0,s.Z)(e);i.current({google_open_id_token:e,autologin:!1},{google_open_id_token:e,first_name:n.given_name,last_name:n.family_name,image_url:n.picture})};return(async()=>{await(null==n?void 0:n.ensureInit("button",e)),(0,b.ru)({buttonRef:t,isDesktop:!1})})(),()=>{null==n||n.removeCallback(e)}}),[n]),(0,u.jsx)(d.xu,{position:"relative",children:(0,u.jsx)(d.iP,{onTap:()=>(0,f.My)("gsi_personalized_button.tap"),children:(0,u.jsx)(d.xu,{height:44,children:(0,u.jsx)(d.xu,{ref:t,position:"absolute",width:"100%"})})})})}function w(e){var t;const{checkExperiment:n}=(0,_.Y)(),{anyEnabled:o}=n("mweb_google_disabled"),i=(0,g.Z)();return(null===(t=(0,c.k6)().location.state)||void 0===t?void 0:t.isAppFactoryOAuth)?null:o?(0,u.jsx)(h,{}):i?(0,u.jsx)(y,{onGoogleConnectSuccess:e.onGoogleConnectSuccess}):(0,u.jsx)(x,{...e})}},738759:(e,t,n)=>{n.d(t,{Z:()=>z,R:()=>j});var o=n(667294),i=n(966338),s=n(157796),r=n(391254),a=n(883119),l=n(100183);const c={yellow:"#FAB904",orange:"#FF6400",green:"#0FA573",blue:"#0074e8",successBase:"#B469EB"},d="DelightfulBubbles__",u=[`${d}fadeIn {\n  to {\n    opacity: 1;\n  }\n}\n`,`${d}growShrink {\n  0% {\n    transform: scale(0);\n  }\n  50% {\n    transform: scale(1);\n  }\n  100% {\n    transform: scale(0);\n  }\n}\n`,`${d}swirl {\n  0% {\n    transform: rotate(0deg) translateX(-50%);\n  }\n  100% {\n    transform: rotate(-360deg) translateX(-50%);\n  }\n}\n`,...Object.keys(c).map((e=>{const t=((e,t)=>{const n=e.length,o=e.indexOf(t)+1>=n?0:e.indexOf(t)+1;return e[o]})(Object.keys(c),e);return`${d}colorChange_${e} {\n      0% {\n        background-color: ${c[e]};\n      }\n      50% {\n        background-color: ${c[t]};\n      }\n    }\n    `}))],h=(0,l.Ll)(u);var g=n(785893);function _(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}const p=e=>{const{delay:t,diameter:n,height:o,initialColor:i,width:s,xOffset:r,yOffset:l}=e,u=((e,t,n,o,i,s,r,a)=>({bubblePositionStyles:{position:"absolute",marginLeft:s/2-i+"px",marginTop:n/2-i+"px",left:`${r}px`,top:`${a}px`},bubbleSwirlStyles:{animation:`${d}swirl 6s ease-in-out ${e}s infinite forwards`,WebkitAnimation:`${d}swirl 6s ease-in-out ${e}s infinite forwards`},bubbleGrowShrinkStyles:{transform:"scale(0)",WebkitTransform:"scale(0)",animation:`${d}growShrink 6s ease-in-out ${e}s infinite alternate`,WebkitAnimation:`${d}growShrink 6s ease-in-out ${e}s infinite alternate`},bubbleShapeStyles:{height:`${t}px`,width:`${t}px`,borderRadius:"50%",backgroundColor:c[o],opacity:0,animation:`${d}colorChange_${o} 12s steps(1, start) ${e}s infinite forwards,\n      ${d}fadeIn 3s linear ${e}s 1 forwards`,WebkitAnimation:`${d}colorChange_${o} 12s steps(1, start) ${e}s infinite forwards,\n      ${d}fadeIn 3s linear ${e}s 1 forwards`}}))(t,n,o,i,Math.floor(n/2),s,r,l);return(0,g.jsx)(a.xu,{dangerouslySetInlineStyle:{__style:u.bubblePositionStyles},children:(0,g.jsx)(a.xu,{dangerouslySetInlineStyle:{__style:u.bubbleSwirlStyles},children:(0,g.jsx)(a.xu,{dangerouslySetInlineStyle:{__style:u.bubbleGrowShrinkStyles},children:(0,g.jsx)(a.xu,{dangerouslySetInlineStyle:{__style:u.bubbleShapeStyles}})})})})};class b extends o.Component{constructor(...e){super(...e),_(this,"state",{isInitialized:!1})}componentDidMount(){window.requestAnimationFrame((()=>{this.setState({isInitialized:!0})}))}shouldComponentUpdate(e,t){return!e.numBubbles===this.props.numBubbles||!t.isInitialized===this.state.isInitialized}getSize(){const{width:e,height:t}=this.props;return void 0!==e&&void 0!==t?{width:e,height:t}:this.containerElement?{height:this.containerElement.offsetHeight,width:this.containerElement.offsetWidth}:{height:void 0,width:void 0}}createBubbles(e,t){const{bubbleSize:n,delayFactor:o,delayOffset:i,numBubbles:s,xRadiusAdjustment:r,yRadiusAdjustment:a}=this.props,d=e+2*a,u=t+2*r,h=[],_=Object.keys(c);for(let c=0;c<s;c+=1){const s=n+(0,l.XF)(-4,4),r=_[c%_.length],{x:a,y:b}=(0,l.CC)(d,u);h.push((0,g.jsx)(p,{delay:c*o+i,diameter:s,initialColor:r,xOffset:a,yOffset:b,height:e,width:t},c))}return h}render(){const{height:e,width:t}=this.getSize();return(0,g.jsxs)(a.xu,{height:e,width:t,position:"relative",children:[(0,g.jsx)(r.Z,{unsafeCSS:h}),(0,g.jsxs)(a.xu,{position:"absolute",ref:e=>this.containerElement=e,children:[this.state.isInitialized&&this.createBubbles(e||0,t||0),(0,g.jsx)(a.xu,{position:"relative",children:this.props.children})]})]})}}_(b,"defaultProps",{bubbleSize:16,numBubbles:10,xRadiusAdjustment:0,yRadiusAdjustment:0,delayFactor:1,delayOffset:0});const f=b;var m=n(7111),x=n(705834),y=n(898781),w=n(70523),v=n(693099);const j=e=>new Set([...v.EU,...v.rT,...v.IO]).has(e),A={default:3e3,email:3e3,facebook:5e3,google:4e3,gtap:4500};function z({isReducedHeader:e,platform:t}){const n=(0,y.ZP)(),r=(0,i.useDispatch)(),l=(0,i.useSelector)((({session:e})=>e.country));(0,o.useEffect)((()=>{"gtap"!==t&&r((0,w.e6)())}),[]);const c=j(l);return(0,g.jsx)(o.Fragment,{children:c?(0,g.jsx)(x.Z,{viewType:10,viewParameter:259,children:(0,g.jsxs)(a.xu,{position:"fixed",top:!0,left:!0,right:!0,bottom:!0,width:"100%",color:"default",children:[(0,g.jsx)(s.Z,{color:"dark",duration:A[t],finalProgress:95}),(0,g.jsxs)(a.kC,{alignItems:"stretch",justifyContent:"start",direction:"column",height:"100%",children:[(0,g.jsxs)(a.xu,{margin:10,children:[(0,g.jsx)(a.X6,{overflow:"normal",size:e?"500":"600",children:n._('Your account is being created!', 'account created heading', 'account created heading')}),(0,g.jsx)(a.xv,{overflow:"normal",children:n._('You can now save ideas, get personalized recommendations, and more', 'loading state description while waiting for any sign up registration to complete', 'loading state description while waiting for any sign up registration to complete')})]}),(0,g.jsx)(a.kC,{alignItems:"center",flex:"grow",justifyContent:"center",children:(0,g.jsx)(f,{delayOffset:-5,xRadiusAdjustment:40,yRadiusAdjustment:40,children:(0,g.jsx)(a.xu,{alignItems:"center",color:"lightWash",direction:"column",display:"flex",height:120,justifyContent:"end",overflow:"hidden",rounding:"circle",width:120,children:(0,g.jsx)(a.JO,{accessibilityLabel:n._('Loading', 'Loading state for any sign up registration', 'Loading state for any sign up registration'),color:"subtle",icon:"person",size:"90"})})})})]})]})}):(0,g.jsx)(m.Z,{})})}},721809:(e,t,n)=>{n.d(t,{Z:()=>a});var o=n(121151),i=n(898781),s=n(883119),r=n(785893);function a({type:e,onDismiss:t}){const n=(0,i.ZP)(),a="facebook"===e?n._('Oops! Facebook isn\'t available', 'socialAuthDisabled.facebook.title', 'Title for the modal shown when Facebook services are not working'):n._('Oops! Google isn\'t available', 'socialAuthDisabled.google.title', 'Title for the modal shown when Google services are not working'),l="facebook"===e?n._('Looks like Facebook isn\'t available right now.\n\nGive another option a try or refresh the page and try again later.', 'socialAuthDisabled.facebook.description', 'Description for the modal shown when Facebook services are not working'):n._('Looks like Google isn\'t available right now.\n\nGive another option a try or refresh the page and try again later.', 'socialAuthDisabled.google.description', 'Description for the modal shown when Google services are not working');return(0,r.jsx)(o.ZP,{accessibilityModalLabel:"",mobileHideCloseIcon:!0,isOpen:!0,onDismiss:t,type:"social_auth_disabled_modal",children:(0,r.jsxs)(s.xu,{padding:5,children:[(0,r.jsx)(s.xu,{marginBottom:3,children:(0,r.jsx)(s.X6,{size:"500",accessibilityLevel:1,children:a})}),(0,r.jsx)(s.xu,{marginBottom:3,children:(0,r.jsx)(s.xv,{children:l})}),(0,r.jsx)(s.kC,{alignItems:"stretch",justifyContent:"start",direction:"column",children:(0,r.jsx)(s.xu,{paddingY:1,children:(0,r.jsx)(s.zx,{accessibilityLabel:n._('OK', 'socialAuthDisabled.okButton', 'Button that closes modal shown when Facebook or Google services are not working'),size:"lg",color:"red",text:n._('OK', 'socialAuthDisabled.okButton', 'Button that closes modal shown when Facebook or Google services are not working'),onClick:t})})})]})})}},169347:(e,t,n)=>{n.d(t,{h:()=>r,r:()=>s});const o=/^(0[1-9]|1[012])\/(0[1-9]|[12][0-9]|3[01])\/((19|20)[0-9]{2})$/,i=/^(19|20)[0-9]{2}$/,s=e=>{if(o.test(e)||i.test(e)){return new Date(e).getTime()/1e3}return-1},r=(e,t)=>{const n=new Date;if(o.test(e)){const o=new Date(n.getFullYear()-t,n.getMonth(),n.getDate());return new Date(e)<=o}if(i.test(e)){const o=n.getFullYear()-t-1;return new Date(e).getFullYear()<=o}return!1}},760547:(e,t,n)=>{n.d(t,{m:()=>S});var o=n(966338),i=n(231486),s=n(957161),r=n(693099);function a(){return function(e){const t={};return e.forEach((e=>{const n=JSON.parse((0,s.qn)(e));n&&(t[e]=n)})),JSON.stringify(t)}([r.E9])}var l=n(6637),c=n(729252),d=n(371415),u=n(383690),h=n(743413),g=n(45169),_=n(81759),p=n(957191),b=n(983360),f=n(226198),m=n(622096),x=n(738759),y=n(867820),w=n(408285),v=n(584595),j=n(396910),A=n(330102),z=n(297507),k=n(3223),O=n(969230);const S=()=>{const e=(0,o.useDispatch)(),{country:t,country_from_hostname:n,country_from_ip:s,region_from_ip:r,isAuthenticated:S,origin:C,inviteCode:I,locale:D,referrer:P}=(0,o.useSelector)((({session:e})=>e)),G=(0,u.Z)();return o=>{G();let u=0;const T=(0,x.R)(t),$=new Promise(((e,t)=>{T?setTimeout((()=>{e()}),3e3):e()}));return new Promise(((x,G)=>{const T=()=>{const B=S,R=(0,_.t_)();((0,j.lJ)(C)&&!1?(0,j.$p)({invite_code:I,locale:D,referrer:P,...o}):l.ZP.create("UserRegisterResource",{...o,get_user:!0,user_behavior_data:a(),visited_pages:R}).callCreate()).then((i=>$.then((()=>{var a,l,u;(0,m.L_)("d_pif_invite"),e((a=i.resource_response.data.user,l=i.client_context.active_experiments,u=i.client_context.triggerable_experiments,{type:b.c7,payload:{user:a,active:l,triggerable:u}})),e((0,p.OD)(i.client_context.placed_experiences)),B&&(0,d.Z)("registration.signup_success");const _=o.email||o.username||null,f=o.password||null;if(navigator.credentials&&_&&f)try{const e=new window.PasswordCredential({id:_,password:f});(0,y.My)("unauth_mweb.navigatorCredentials.store.attempt"),navigator.credentials&&"function"==typeof navigator.credentials.store&&navigator.credentials.store(e).then((()=>{(0,y.My)("unauth_mweb.navigatorCredentials.store.success")}))}catch(v){(0,y.My)("unauth_mweb.navigatorCredentials.store.error")}(0,O.h3)({action:"signup",type:(0,g.sR)(o)}),(0,c.Z)(),B&&(0,w.Dm)(),(0,k.c_)({country:t,countryFromHostName:n,countryFromIp:s})&&(0,z.yP)({id:h.J3,eventCategory:"NewUsers",eventName:"Mobile"}),(0,k.$r)({country:t,countryFromHostName:n,countryFromIp:s,regionFromIp:r})&&(0,z.jw)({id:"flashtalking-m-reg",eventCategory:"NewUsers",eventName:"Mobile"}),(0,k.NR)(t,r)&&(0,z.EN)({pixelId:A.NS,eventCategory:"NewUsers",eventName:"Mobile"}),(0,k.Fc)({country:t,countryFromHostName:n,countryFromIp:s})&&(0,z.Be)({pixelId:A.Ks,eventCategory:"NewUsers",eventName:"Mobile"}),x()})))).catch((e=>{const{apiErrorCode:t=null}=e;t&&t===v.tz&&((0,y.My)("multi_step_set_age_restrict_cookie"),i.t8(f.It,"1",(0,i.kZ)(f.It))),u<3&&(!t||[v.dO,v.cZ].includes(t))?(u+=1,T()):((0,O.wp)({action:"signup",type:(0,g.sR)(o)}),B&&(0,d.Z)("registration.signup_error"),G(e))}))};T()}))}}},634123:(e,t,n)=>{n.d(t,{Z:()=>r});var o=n(56641),i=n(925927),s=n(50286);function r(){const e=(0,i.k6)(),t=(0,s.HG)();return n=>{t?(0,o.Z)(n):e.push(n)}}},472644:(e,t,n)=>{n.d(t,{Z:()=>i});var o=n(785893);function i({size:e}){return(0,o.jsx)("svg",{height:e,viewBox:"0 0 512 512",width:e,xmlns:"http://www.w3.org/2000/svg",children:(0,o.jsxs)("g",{fill:"none",fillRule:"evenodd",children:[(0,o.jsx)("path",{d:"M482.56 261.36c0-16.73-1.5-32.83-4.29-48.27H256v91.29h127.01c-5.47 29.5-22.1 54.49-47.09 71.23v59.21h76.27c44.63-41.09 70.37-101.59 70.37-173.46z",fill:"#4285f4"}),(0,o.jsx)("path",{d:"M256 492c63.72 0 117.14-21.13 156.19-57.18l-76.27-59.21c-21.13 14.16-48.17 22.53-79.92 22.53-61.47 0-113.49-41.51-132.05-97.3H45.1v61.15c38.83 77.13 118.64 130.01 210.9 130.01z",fill:"#34a853"}),(0,o.jsx)("path",{d:"M123.95 300.84c-4.72-14.16-7.4-29.29-7.4-44.84s2.68-30.68 7.4-44.84V150.01H45.1C29.12 181.87 20 217.92 20 256c0 38.08 9.12 74.13 25.1 105.99l78.85-61.15z",fill:"#fbbc05"}),(0,o.jsx)("path",{d:"M256 113.86c34.65 0 65.76 11.91 90.22 35.29l67.69-67.69C373.03 43.39 319.61 20 256 20c-92.25 0-172.07 52.89-210.9 130.01l78.85 61.15c18.56-55.78 70.59-97.3 132.05-97.3z",fill:"#ea4335"}),(0,o.jsx)("path",{d:"M20 20h472v472H20V20z"})]})})}},711781:(e,t,n)=>{n.d(t,{Z:()=>o});const o=e=>{return JSON.parse(decodeURIComponent(atob((t=e.split(".")[1],(t||"").replace(/-/g,"+").replace(/_/g,"/"))).split("").map((e=>"%"+("00"+e.charCodeAt(0).toString(16)).slice(-2))).join("")));var t}},70523:(e,t,n)=>{n.d(t,{e6:()=>i,lW:()=>o,oo:()=>s});const o=()=>({type:"GOOGLE_ONE_TAP_INITIALIZED",payload:Object.freeze({})}),i=()=>({type:"HIDE_GOOGLE_ONE_TAP"}),s=()=>({type:"GOOGLE_ONE_TAP_DISABLED"})}}]);
//# sourceMappingURL=https://sm.pinimg.com/webapp/7740-8dce9ae63d43febc.mjs.map
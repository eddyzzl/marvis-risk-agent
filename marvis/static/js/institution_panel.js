import { api as defaultApi } from './api.js';

const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function createInstitutionPanel({container,api=defaultApi,onIdentityChange=()=>{}}) {
  let settings=null, busy=false, generation=0, destroyed=false;
  container.innerHTML=`<div class="governed-capabilities-panel">
    <div class="governed-capabilities-header"><strong>身份与资源授权</strong><button type="button" data-action="refresh">刷新</button></div>
    <p data-state>正在读取身份模式…</p><p>本地模式保持原工作流。机构模式采用受信签发方的短期签名登录；这里的参考协议不代表外部机构已接入。</p>
    <details open><summary>登录身份</summary><div class="governed-capabilities-grid"><label>登录签发方<input data-input="login-issuer"></label><label>登录主体<input data-input="login-subject"></label></div><button type="button" data-action="challenge">获取登录挑战</button><textarea readonly data-input="challenge" rows="2" aria-label="本次登录挑战"></textarea><label>签名登录凭证（仅本次使用）<textarea data-input="assertion" autocomplete="off" rows="3" placeholder="粘贴签发方返回的 assertion 与 signature"></textarea></label>
    <div class="governed-capabilities-actions"><button type="button" data-action="login">登录</button><button type="button" data-action="logout">退出</button></div></details>
    <details><summary>管理员 · 受信身份与模式</summary><label>本地管理员确认令牌<input data-input="admin" type="password" autocomplete="off"></label>
    <div class="governed-capabilities-grid"><label>签发方<input data-input="issuer"></label><label>密钥标识<input data-input="key-id"></label><label>Ed25519 公钥<input data-input="public-key"></label></div><button type="button" data-action="trust">信任签发方</button>
    <div class="governed-capabilities-grid"><label>主体标识<input data-input="subject"></label><label>显示名称<input data-input="display-name"></label><label>服务端角色<select data-input="role"><option value="maker">创建者</option><option value="checker">复核者</option><option value="admin">管理员</option><option value="member">成员</option><option value="service">服务身份</option></select></label></div><button type="button" data-action="provision">登记主体</button>
    <div class="governed-capabilities-grid"><label>登录受众<input data-input="audience"></label><label>身份模式<select data-input="mode"><option value="trusted_local">本地受信模式</option><option value="institutional">机构受治理模式</option></select></label></div><button type="button" data-action="mode">确认切换模式</button><div class="governed-capabilities-grid"><label>撤销类型<select data-input="revoke-kind"><option value="subject">主体</option><option value="issuer">签发密钥</option><option value="session">登录会话</option></select></label><label>准确撤销 ID<input data-input="revoke-id"></label><label>签发密钥标识（仅密钥撤销）<input data-input="revoke-key"></label></div><label><input type="checkbox" data-input="revoke-approved">确认撤销此准确身份凭证</label><button type="button" data-action="revoke-identity">确认撤销身份凭证</button></details>
    <details><summary>资源授权 · 版本绑定</summary><div class="governed-capabilities-grid"><label>资源类型<select data-input="kind"><option value="task">任务</option><option value="dataset">数据集</option><option value="artifact">产物</option><option value="plan">计划</option></select></label><label>准确资源 ID<input data-input="resource"></label><label>目标主体 ID<input data-input="target"></label><label>当前授权版本<input data-input="revision" type="number" min="0" value="0"></label></div>
    <div data-actions>${['read','write','execute','share','approve'].map((a)=>`<label><input type="checkbox" value="${a}">${({read:'读取',write:'修改',execute:'执行',share:'授权他人',approve:'人工复核'})[a]}</label>`).join('')}</div>
    <div class="governed-capabilities-actions"><button type="button" data-action="grant">确认授权</button><button type="button" data-action="revoke-grant">撤销授权</button><button type="button" data-action="owner">管理员绑定既有资源所有者</button></div></details>
    <div data-evidence></div><p data-error role="status"></p></div>`;
  const input=(name)=>container.querySelector(`[data-input="${name}"]`);
  const value=(name)=>input(name).value.trim();
  const error=(text)=>{if(destroyed)return;container.querySelector('[data-error]').textContent=text;};
  const headers=()=>({'x-marvis-plugin-admin':value('admin')});
  const request=(path,body,method='POST')=>api(path,{method,headers:headers(),body:JSON.stringify(body)});
  async function refresh() {
    const current=++generation;
    const status=await api('/api/institution/status');
    let me=null;
    if(status.mode==='institutional') {try {me=await api('/api/institution/me');} catch (e) {if(Number(e.status)!==401) throw e;}}
    if(destroyed||current!==generation) return;
    settings=status;
    input('mode').value=status.mode;input('audience').value=status.audience;
    container.querySelector('[data-state]').textContent=`${status.mode==='institutional'?'机构受治理':'本地受信'} · 配置版本 ${status.revision}${me?` · ${me.display_name} · ${me.role}`:' · 当前未登录机构身份'}`;
    container.querySelector('[data-evidence]').innerHTML=`<small>协议 ${escape(status.protocol)} · ${escape(status.algorithm)}。任意 Notebook、本地路径直接注册及尚未接通的入口会明确拒绝；服务身份不能执行人工复核。</small>`;
  }
  async function action(name) {
    if(busy||destroyed) return;
    busy=true;error('');
    for(const b of container.querySelectorAll('button')) b.disabled=true;
    try {
      let result;
      if(name==='refresh') return await refresh();
      if(name==='challenge') {result=await request('/api/institution/challenge',{issuer:value('login-issuer'),subject:value('login-subject')});input('challenge').value=JSON.stringify(result);}
      else if(name==='revoke-identity') {if(!input('revoke-approved').checked) throw new Error('请先确认准确撤销的身份凭证。');result=await request('/api/institution/revoke',{kind:value('revoke-kind'),resource_id:value('revoke-id'),key_id:value('revoke-kind')==='issuer'?value('revoke-key'):null});input('revoke-approved').checked=false;}
      else if(name==='login') {result=await request('/api/institution/login',JSON.parse(input('assertion').value));input('assertion').value='';input('challenge').value='';await onIdentityChange(result);}
      else if(name==='logout') {result=await request('/api/institution/logout',{});await onIdentityChange(null);}
      else if(name==='trust') result=await request('/api/institution/issuers',{issuer:value('issuer'),key_id:value('key-id'),public_key:value('public-key')});
      else if(name==='provision') result=await request('/api/institution/subjects',{issuer:value('issuer'),subject:value('subject'),display_name:value('display-name'),role:value('role')});
      else if(name==='mode') {result=await request('/api/institution/mode',{mode:value('mode'),audience:value('audience'),expected_revision:settings.revision},'PUT');await onIdentityChange(null);}
      else {
        const binding={kind:value('kind'),resource_id:value('resource'),expected_revision:Number(value('revision'))};
        if(name==='owner') result=await request('/api/institution/owners',{...binding,owner_id:value('target')});
        else if(name==='grant') result=await request('/api/institution/grants',{...binding,subject_id:value('target'),actions:Array.from(container.querySelectorAll('[data-actions] input:checked')).map((x)=>x.value)});
        else if(name==='revoke-grant') result=await request('/api/institution/grants/revoke',{...binding,subject_id:value('target')});
        else return;
        if(result.revision!==undefined) input('revision').value=String(result.revision);
      }
      if(destroyed) return;
      await refresh();error(`已完成 ${name}${result?.id?` · ${result.id}`:''}`);
    } catch(e) {error(e.detail?.code||e.detail||e.message||'操作未完成，请核对身份与版本后重试。');}
    finally {busy=false;for(const b of container.querySelectorAll('button')) b.disabled=false;}
  }
  const click=(event)=>{const button=event.target.closest('[data-action]');if(button&&container.contains(button)) void action(button.dataset.action);};
  container.addEventListener('click',click);
  const ready=refresh().catch((e)=>error(e.message));
  return {ready,refresh,destroy(){destroyed=true;++generation;container.removeEventListener('click',click);input('assertion').value='';input('challenge').value='';input('admin').value='';}};
}

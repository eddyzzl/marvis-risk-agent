import { api as defaultApi } from './api.js';

const escape=(value)=>String(value??'').replace(/[&<>"']/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function createInstanceSynchronizationPanel({container,api=defaultApi,getTaskId=()=>null}) {
  let catalog=null,bundle=null,busy=false,generation=0,destroyed=false,revocation=null,fileGeneration=0;
  container.innerHTML=`<div class="governed-capabilities-panel"><div class="governed-capabilities-header"><strong>实例间资源交接</strong><button type="button" data-action="refresh">刷新</button></div>
    <p>按受信节点签名交接准确模型、策略和 Workflow 版本；接收不改变当前生产部署。模型目前支持原生二分类 LR 的完整预处理管线，其他算法明确拒绝。</p>
    <label>本地管理员确认令牌<input type="password" autocomplete="off" data-input="admin"></label><p data-state>正在读取实例身份…</p>
    <details><summary>节点信任</summary><label>对端节点 ID<input data-input="node-id"></label><label>对端 Ed25519 公钥<input data-input="public-key"></label><div class="governed-capabilities-actions"><button type="button" data-action="trust">信任节点</button><button type="button" data-action="review-revoke">查看撤销影响</button><button type="button" data-action="revoke">确认撤销所列版本</button></div><div data-revocation>先读取撤销提案；准确交接头版本变化后必须重新核对。</div><label><input type="checkbox" data-input="revoke-approved">确认停用所列版本及节点</label></details>
    <details><summary>发送准确版本</summary><div class="governed-capabilities-grid"><label>资源类型<select data-input="kind"><option value="model">模型</option><option value="strategy">策略</option><option value="workflow">Workflow</option></select></label><label>原生资源 ID<input data-input="native-ref"></label><label>交接名称<input data-input="name"></label><label>版本<input data-input="version" value="1.0.0"></label><label>接收节点 ID<input data-input="receiver"></label></div><button type="button" data-action="export">生成已签名交接文件</button><a data-download hidden>下载交接文件</a></details>
    <details><summary>接收与权限确认</summary><label>签名交接 JSON 文件<input type="file" accept=".json,application/json" data-input="file"></label><div data-bundle>请选择已签名交接文件。</div><label>当前接收版本<input data-input="revision" type="number" min="0" value="0"></label><label><input type="checkbox" data-input="approved">确认所列权限及当前任务接收位置</label><button type="button" data-action="import">确认接收</button></details>
    <div data-versions></div><p data-error role="status"></p></div>`;
  const input=(n)=>container.querySelector(`[data-input="${n}"]`),value=(n)=>input(n).value.trim();
  const headers=()=>({'x-marvis-plugin-admin':value('admin')});
  const error=(s)=>{if(destroyed)return;container.querySelector('[data-error]').textContent=s;};
  const request=(path,body)=>api(path,{method:'POST',headers:headers(),body:JSON.stringify(body)});
  let downloadUrl=null;
  async function refresh() {
    const current=++generation;
    const identity=await api('/api/institution/status');
    if(identity.mode==='trusted_local'&&!value('admin')) {
      if(!destroyed&&current===generation) container.querySelector('[data-state]').textContent='填写本地管理员确认令牌后刷新；受保护版本信息尚未读取。';
      return;
    }
    const data=await api('/api/instance-sync',{headers:headers()});
    if(destroyed||current!==generation) return;
    catalog=data;
    container.querySelector('[data-state]').textContent=`本实例 ${data.node.id} · 公钥 ${data.node.public_key}`;
    container.querySelector('[data-versions]').innerHTML=data.versions.map((v)=>`<div class="governed-capabilities-version"><strong>${escape(v.kind)} · ${escape(v.resource_name)} ${escape(v.version)}</strong><small>${v.active?'当前交接版本':'保留版本'} · 头版本 ${v.revision}${v.node_revoked?' · 节点已撤销':''}</small><small>准确权限：${escape((v.permissions||[]).join('、')||'无新增权限')}</small><label><input type="checkbox" data-version-approval="${escape(v.id)}">确认此版本权限及回滚头</label><div class="governed-capabilities-actions"><button type="button" data-action="readback" data-id="${escape(v.id)}">核验原生产物</button><button type="button" data-action="rollback" data-id="${escape(v.id)}" ${v.active||v.node_revoked?'disabled':''}>回滚到此版本</button></div></div>`).join('')||'<small>尚未接收资源版本。</small>';
  }
  async function selectFile() {
    const selected=++fileGeneration;bundle=null;input('approved').checked=false;
    try {const file=input('file').files?.[0];if(!file) return;if(file.size>8*1024*1024) throw new Error('交接文件超过 8 MiB。');const parsed=JSON.parse(await file.text());if(destroyed||selected!==fileGeneration)return;bundle=parsed;const body=bundle.envelope.body;container.querySelector('[data-bundle]').textContent=`${body.kind} · ${body.name} ${body.version} · 发送节点 ${body.origin_node} · 权限 ${body.permissions.join('、')||'无新增权限'}。服务端仍将验证签名、准确内容与依赖。`;}
    catch(e){bundle=null;error(e.message);}
  }
  async function action(name,id) {
    if(busy||destroyed) return;
    busy=true;error('');
    try {
      let result;
      if(name==='refresh') return await refresh();
      if(name==='trust') result=await request('/api/instance-sync/trust',{id:value('node-id'),public_key:value('public-key')});
      else if(name==='review-revoke') {
        const selected=value('node-id');
        result=await api(`/api/instance-sync/nodes/${encodeURIComponent(selected)}/revocation-proposal`,{headers:headers()});
        if(destroyed||value('node-id')!==selected) return;
        revocation=result;input('revoke-approved').checked=false;
        container.querySelector('[data-revocation]').textContent=result.heads.map(h=>`${h.kind} · ${h.resource_name} · 头 ${h.revision} · ${h.version_id}`).join('；')||'此节点尚无已接收版本。';
      }
      else if(name==='revoke') {
        if(!revocation||revocation.node_id!==value('node-id')||!input('revoke-approved').checked) throw new Error('先查看准确撤销影响并确认所列版本。');
        result=await request(`/api/instance-sync/nodes/${encodeURIComponent(revocation.node_id)}/revoke`,{expected_heads:revocation.heads,proposal_hash:revocation.proposal_hash});
        revocation=null;input('revoke-approved').checked=false;
      }
      else if(name==='export') {result=await request('/api/instance-sync/export',{kind:value('kind'),native_ref:value('native-ref'),name:value('name'),release_version:value('version'),receiver_node:value('receiver')});if(downloadUrl) URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(new Blob([JSON.stringify(result)],{type:'application/json'}));const link=container.querySelector('[data-download]');link.href=downloadUrl;link.download=`${value('name')}-${value('version')}.json`;link.hidden=false;}
      else if(name==='import') {if(!bundle||!input('approved').checked) throw new Error('请选择交接文件并确认所列权限。');const task=getTaskId();if(!task) throw new Error('请先选择接收任务。');result=await request('/api/instance-sync/import',{...bundle,task_id:task,expected_revision:Number(value('revision')),approved_permissions:bundle.envelope.body.permissions});if(!destroyed&&getTaskId()===task) input('revision').value=String(result.revision);}
      else if(name==='readback') result=await api(`/api/instance-sync/versions/${encodeURIComponent(id)}/readback`,{headers:headers()});
      else if(name==='rollback') {const version=catalog.versions.find((v)=>v.id===id);
        if(!version||!container.querySelector(`[data-version-approval="${id}"]`)?.checked) throw new Error('先确认此准确版本的权限和回滚头。');
        result=await request('/api/instance-sync/rollback',{version_id:id,expected_revision:version.revision,approved_permissions:version.permissions});}
      else return;
      if(destroyed) return;
      await refresh();error(`已完成 ${name}${result?.native_ref?` · 原生资源 ${result.native_ref}`:''}`);
    }catch(e){error(e.detail?.code||e.detail||e.message||'操作未完成；保留当前输入以核对版本。');}
    finally {busy=false;}
  }
  const click=(event)=>{const b=event.target.closest('[data-action]');if(b&&container.contains(b)) void action(b.dataset.action,b.dataset.id);};
  const change=(event)=>{if(event.target===input('file')) void selectFile();};
  container.addEventListener('click',click);container.addEventListener('change',change);
  const ready=refresh().catch((e)=>error(e.detail?.code||e.message));
  return {ready,refresh,destroy(){destroyed=true;++generation;++fileGeneration;container.removeEventListener('click',click);container.removeEventListener('change',change);input('admin').value='';bundle=null;if(downloadUrl) URL.revokeObjectURL(downloadUrl);}};
}

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_role_specific_blind_panel_does_not_show_checker_metrics_to_maker():
    subprocess.run(['node','--input-type=module','-e', '''
        import assert from 'node:assert/strict';
        import { blindOotHtml, holdoutError } from './marvis/static/js/blind-oot-controller.js';
        const summary = {state:'committed', viewer_role:'develop',freeze_hash:'0'.repeat(64)};
        const secret = {metrics:{auc:0.912345,ks:0.45}};
        const maker = blindOotHtml({summary, principal:{role:'maker'},result:secret});
        assert(!maker.includes('0.912345'));
        assert(maker.includes('开发人员和 Agent 不接收'));
        const checker = blindOotHtml({summary:{...summary,viewer_role:'evaluate'},principal:{role:'checker'},result:secret});
        assert(checker.includes('0.912345'));
        const frozen = blindOotHtml({summary:{...summary,state:'frozen',viewer_role:'evaluate'},principal:{role:'checker'}});
        assert(frozen.includes('评估封存集一次'));
        const escaped = blindOotHtml({summary:null,principal:{role:'admin'},participants:[{id:'id',role:'maker',display_name:'<script>secret</script>'}]});
        assert(!escaped.includes('<script>'));
        assert(escaped.includes('&lt;script&gt;'));
        assert(holdoutError({detail:{code:'already_exposed_source_cannot_be_blind'}}).includes('不能重新认定'));
    '''],cwd=ROOT,capture_output=True,text=True,check=True)


def test_blind_mutations_send_json_over_actual_http_wire():
    subprocess.run(['node','--input-type=module','-e', '''
        import assert from 'node:assert/strict';
        import http from 'node:http';
        import { createBlindOotController } from './marvis/static/js/blind-oot-controller.js';
        const posts = [], listeners = new Map();
        let view = 'develop', state = 'sealed';
        const server = http.createServer(async (req,res) => {
          let raw=''; for await (const chunk of req) raw+=chunk;
          res.setHeader('Content-Type','application/json');
          if (req.method==='POST') {
            assert.match(req.headers['content-type'], /application\\/json/);
            posts.push({path:req.url,body:JSON.parse(raw)});
            res.end('{}'); return;
          }
          if (req.url.endsWith('/me')) res.end(JSON.stringify({role:view==='develop'?'maker':view==='evaluate'?'checker':'admin'}));
          else if (req.url.endsWith('/candidates')) res.end(JSON.stringify({candidates:[{artifact_id:'native',algorithm:'lr'}]}));
          else res.end(JSON.stringify({viewer_role:view,state,freeze_hash:'0'.repeat(64),recovery_options:['resume_before_read']}));
        });
        await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
        global.document={baseURI:`http://127.0.0.1:${server.address().port}/`,body:{dataset:{}}};
        const NativeFormData=global.FormData;
        global.FormData=class extends NativeFormData { constructor(form) { super(); for(const [k,v] of Object.entries(form?.values||{})) this.append(k,v); } };
        const root={hidden:false,innerHTML:'',addEventListener:(type,handler)=>listeners.set(type,handler)};
        const controller=createBlindOotController({root,getTask:()=>({id:'task',task_type:'modeling'}),openIdentity:()=>{}});
        const wait=async n=>{for(let i=0;i<100&&posts.length<n;i++) await new Promise(resolve=>setTimeout(resolve,10)); assert.equal(posts.length,n); await new Promise(resolve=>setTimeout(resolve,30));};
        try {
          await controller.refresh();
          const form={dataset:{blindForm:'freeze'},reportValidity:()=>true,values:{artifact_id:'native'}};
          listeners.get('submit')({preventDefault(){},target:{closest:()=>form}});
          await wait(1); assert.deepEqual(posts[0].body,{artifact_id:'native'});
          view='evaluate'; state='frozen'; await controller.refresh({force:true});
          listeners.get('click')({target:{closest:()=>({dataset:{blindAction:'evaluate'}})}});
          await wait(2); assert.equal(posts[1].body.freeze_hash,'0'.repeat(64));
          view='custody'; state='uncertain'; await controller.refresh({force:true});
          const recovery={dataset:{blindForm:'recover'},reportValidity:()=>true,values:{mode:'resume_before_read',reason:'Synthetic safe recovery'}};
          listeners.get('submit')({preventDefault(){},target:{closest:()=>recovery}});
          await wait(3); assert.equal(posts[2].body.mode,'resume_before_read');
        } finally { await new Promise(resolve=>server.close(resolve)); }
    '''],cwd=ROOT,capture_output=True,text=True,check=True,timeout=20)

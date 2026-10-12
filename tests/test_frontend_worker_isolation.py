"""Actual dedicated module through the application's shared fetch wire."""

import subprocess

from test_frontend_governed_capabilities import HARNESS, ROOT


WORKER_HARNESS = (
    HARNESS.replace(
        "import {createGovernedCapabilitiesPanel} from './marvis/static/js/governed_capabilities_panel.js';",
        "import {createWorkerIsolationPanel} from './marvis/static/js/worker_isolation_panel.js';",
    )
    .replace("dataset.capabilityField", "dataset.workerIsolationField")
    .replace("dataset.capabilityAction", "dataset.workerIsolationAction")
    .replace("dataset.capabilityStatus", "dataset.workerIsolationStatus")
)


def test_worker_mode_actual_json_wire_preserves_interpreter_notebook_and_rolls_back_refusal():
    script = (
        WORKER_HARNESS
        + r"""
      const settings={execution_mode:'conda_env',kernel_name:'selected-kernel',conda_env_name:'selected-env',
        python_executable:'/synthetic/python',notebook_network_mode:'loopback_only',notebook_memory_limit_mb:320,
        worker_isolation_mode:'trusted_local'};
      const wire=[];let refuse=false;
      globalThis.fetch=async(url,options={})=>{
        wire.push({url,options});
        assert.equal(new URL(url,'http://127.0.0.1/').pathname,'/api/settings/execution-environment');
        if(options.method==='PUT'){
          assert.equal(typeof options.body,'string');
          const body=JSON.parse(options.body);
          assert.deepEqual(body,{...settings,worker_isolation_mode:field('mode').value});
          if(refuse)return reply({detail:{code:'worker_isolation_unavailable'}},422);
          settings.worker_isolation_mode=body.worker_isolation_mode;
        }
        return reply({settings:{...settings},validation:{worker_capability:{available:true,startup_checked:true}}});
      };
      const panel=createWorkerIsolationPanel({container:root});await panel.ready;
      assert.equal(field('mode').value,'trusted_local');
      field('mode').value='os_sandbox';await action('save').emit('click');
      assert.equal(settings.worker_isolation_mode,'os_sandbox');assert.match(status(),/预检通过/);
      assert.match(root.textContent,/内置平台工具仍/);
      refuse=true;field('mode').value='trusted_local';await action('save').emit('click');
      assert.equal(field('mode').value,'os_sandbox');assert.match(status(),/未保存/);
      assert.equal(wire.filter(call=>call.options.method==='PUT').length,2);
      panel.destroy();assert.equal(root.children.length,0);
    """
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

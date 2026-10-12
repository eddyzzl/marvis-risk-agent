"""Candidate stability keeps exact refs and rejects stale UI responses."""

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).parents[1]


def test_asset_stability_selection_and_task_switch_reject_stale_responses():
    script = r"""
    import assert from 'node:assert/strict';
    import { mountStrategyAssetStabilityPanel } from './marvis/static/js/v2/strategy_asset_stability_panel.js';
    const calls = [];
    globalThis.fetch = (url, options = {}) => new Promise(resolve => {
      calls.push({url, options, resolve});
    });
    const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
    const reply = async (call, value) => {
      call.resolve({ok: true, json: async () => value}); await flush();
    };
    const source = {artifact_id:'source-exact', artifact_content_hash:'source-sha', asset_id:'asset-exact', asset_hash:'asset-sha'};
    const sample = {bundle_artifact_id:'bundle-exact', expected_bundle_artifact_content_hash:'bundle-sha', membership_artifact_id:'membership-exact', expected_membership_artifact_content_hash:'membership-sha', expected_bundle_id:'bundle-id', expected_sample_design_id:'sample-id', expected_sample_design_content_hash:'sample-sha'};
    const catalogue = {sources:[{source_kind:'complete_tree',ref:source}], samples:[{ref:sample,counts:{development:6,validation:2,oot:2}}], evidence:[]};
    class Root {
      constructor() { this.listeners = new Map(); this.nodes = new Map(); }
      set innerHTML(value) {
        this.html = value; this.nodes.clear();
        const body = {textContent:''};
        Object.defineProperty(body,'innerHTML',{set: html => {
          body.html = html;
          const selectors = ['source','sample','freeze','measure','report','refresh','existing-freeze','resume','status','results'];
          for (const name of selectors) {
            const key = `[data-asset-${name}]`;
            const node = {value:'', disabled: name === 'measure' || name === 'report', textContent:'', innerHTML:'', hasAttribute: attr => attr === `data-asset-${name}`, closest: () => node, insertAdjacentHTML: (_position,text) => { node.innerHTML += text; }};
            Object.defineProperty(node,'isConnected',{get: () => this.nodes.get(key) === node});
            this.nodes.set(key,node);
          }
        }});
        this.nodes.set('[data-asset-stability-body]',body);
      }
      querySelector(selector) { return this.nodes.get(selector) || null; }
      querySelectorAll() { return [{dataset:{assetPartition:'validation'}},{dataset:{assetPartition:'oot'}}]; }
      addEventListener(type, callback, options) {
        const callbacks = this.listeners.get(type) || new Set(); callbacks.add(callback); this.listeners.set(type,callbacks);
        options.signal.addEventListener('abort',() => callbacks.delete(callback),{once:true});
      }
      async emit(type, event = {}) { for (const callback of this.listeners.get(type) || []) await callback(event); }
    }
    const root = new Root();
    mountStrategyAssetStabilityPanel(root,'first-task');
    await reply(calls[0],catalogue);
    root.querySelector('[data-asset-source]').value = '0';
    root.querySelector('[data-asset-sample]').value = '0';
    const pending = root.emit('click',{target:root.querySelector('[data-asset-freeze]')});
    assert.deepEqual(JSON.parse(calls[1].options.body),{source_ref:source,sample_design_ref:sample,partitions:['validation','oot']});
    await root.emit('change');
    await reply(calls[1],{ref:{stability_id:'stale-freeze'}}); await pending;
    assert.equal(root.querySelector('[data-asset-measure]').disabled,true);
    assert.equal(root.querySelector('[data-asset-report]').disabled,true);
    assert.equal(root.querySelector('[data-asset-results]').innerHTML,'');
    assert.match(root.querySelector('[data-asset-status]').textContent,/选择已变化/);
    const pendingFresh = root.emit('click',{target:root.querySelector('[data-asset-freeze]')});
    const frozen = {artifact_id:'freeze-exact',artifact_content_hash:'freeze-sha',stability_id:'freeze-id',content_hash:'freeze-content'};
    await reply(calls[2],{ref:frozen}); await pendingFresh;
    assert.equal(root.querySelector('[data-asset-measure]').disabled,false);
    const pendingMeasure = root.emit('click',{target:root.querySelector('[data-asset-measure]')});
    assert.deepEqual(JSON.parse(calls[3].options.body),{frozen_ref:frozen});
    mountStrategyAssetStabilityPanel(root,'second-task');
    await reply(calls[4],catalogue);
    await reply(calls[3],{ref:{stability_id:'first-task-measure'},evidence:{}}); await pendingMeasure;
    assert.equal(root.querySelector('[data-asset-results]').innerHTML,'');
    assert.equal(root.querySelector('[data-asset-measure]').disabled,true);
    assert.equal(calls[4].url,'/api/tasks/second-task/strategy-asset-stability');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

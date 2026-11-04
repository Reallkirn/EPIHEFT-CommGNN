"""Run the fixed Pegasus-S main experiment with a frozen actor or baselines."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
METHODS = ('main', 'HEFT_Comm', 'PEFT', 'CPOP', 'IPPTS_Dyn')
# Also limit the parent process, before NumPy or PyTorch is imported.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'TORCH_NUM_THREADS'):
    os.environ[_key] = '1'


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_one(job):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TORCH_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    # A worker may process several instances; interop is set only once.
    if not getattr(evaluate_one, '_initialized', False):
        torch.set_num_interop_threads(1)
        evaluate_one._initialized = True
    from settings import configs as c
    from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
    from env.workflow_scheduling_v3.simulator_wf import WFEnv

    method, H, idx, device, weights = job
    c.device = device
    c.heartbeat_interval = H
    c.GENindex, c.indEVALindex = 0, idx
    # Rebuild the original type-index file; always use the fixed 30-instance
    # slice, including for --n 1, to preserve arrival sampling and identities.
    full = np.random.RandomState(2024).randint(0, 4, (1, 10000, 30)).astype(np.int64)
    c.valid_dataset = full[:, 30:60]
    seed_all(0)
    c.arr_times = sample_poisson_shape(c.arr_rate, c.valid_dataset.shape)
    seed_all(0)
    actor = batch = None
    if method == 'main':
        import model
        model.device = torch.device(device)
        actor = model.Actor(input_dim_wf=3, input_dim_vm=8, hidden_dim=128,
                            embedding_type='gat', gnn_layers=2, mlp_layers=4,
                            heads=1, dropout=0.)
        actor.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        actor.to(device).eval()
        actor.set_runtime_prior_weight(.10)
        batch = model.BatchGraph(c.normalize)

    seed_all(idx)
    env = WFEnv(c)
    state = env.reset() if actor is not None else env.resetGP()
    bw = np.asarray(env.comm_model.bandwidth_matrix, dtype=np.float64)
    hops = np.asarray(env.comm_model.hop_matrix, dtype=np.int64)
    network_hash = hashlib.sha256(bw.tobytes() + hops.tobytes()).hexdigest()
    fixture_hash = hashlib.sha256(np.asarray(c.valid_dataset[0, idx], dtype=np.int64).tobytes()
                                 + np.asarray(c.arr_times[0, idx], dtype=np.float64).tobytes()).hexdigest()
    start = time.perf_counter()
    decisions = 0
    with torch.inference_mode():
        while True:
            if actor is not None:
                batch.wrapper(*state)
                action = int(actor(state_wf=batch.wf_features, state_vm=batch.vm_features,
                    edge_index_wf=batch.wf_edges, edge_index_vm=batch.vm_edges,
                    mask_wf=batch.wf_masks, mask_vm=batch.vm_masks,
                    batch_wf=batch.wf_batchs, batch_vm=batch.vm_batchs,
                    candidate_task_index=batch.candidate_taskID,
                    state_comm=batch.comm_node_features, edge_index_comm=batch.comm_edge_index,
                    edge_attr_comm=batch.comm_edge_attr, batch_comm=batch.comm_batchs,
                    vm_id_map=batch.vm_id_map, deterministic=True)[0].item())
            elif method == 'HEFT_Comm':
                action = int(env.HEFT())
            elif method == 'PEFT':
                action = int(env.PEFT_comm())
            elif method == 'CPOP':
                action = int(env.CPOP())
            else:
                from env.workflow_scheduling_v3.lib.ippts_dynamic import ippts_dyn_action
                action = int(ippts_dyn_action(env))
            state, _, done = env.step(action) if actor is not None else env.stepGP(action)
            decisions += 1
            if done:
                break
    fct = np.asarray(env.all_flowTime, dtype=np.float64)
    assert env.completedWF == 30 and len(fct) == 30 and np.isfinite(fct).all() and (fct > 0).all()
    return dict(method=method, H=H, instance_id=30+idx, dataset='Pegasus-S', N=10,
                alpha=1.5, p_w=.10 if actor is not None else 0.,
                mean_fct=float(fct.mean()), completed_workflows=30, decision_steps=decisions,
                elapsed_seconds=time.perf_counter()-start, fixture_hash=fixture_hash,
                network_hash=network_hash, checkpoint_sha256=digest_file(weights) if actor is not None else '')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods', default='main,HEFT_Comm', help=','.join(METHODS))
    ap.add_argument('--H', default='0', help='0,3,5,10')
    ap.add_argument('--n', type=int, default=1, help='first n held-out instances, 1..30')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--weights', default=str(ROOT/'main_actor.pth'))
    ap.add_argument('--output', help='new or empty output directory')
    a = ap.parse_args()
    methods = a.methods.split(',')
    try:
        hs = [int(h) for h in a.H.split(',')]
    except ValueError:
        ap.error('H must be comma-separated integers')
    if (not set(methods) <= set(METHODS) or len(set(methods)) != len(methods)
            or not set(hs) <= {0, 3, 5, 10} or len(set(hs)) != len(hs)
            or not 1 <= a.n <= 30 or a.workers < 1):
        ap.error('invalid methods, H, n or workers')
    weights = Path(a.weights).resolve()
    if 'main' in methods and not weights.is_file():
        ap.error('actor checkpoint does not exist: ' + str(weights))
    output = Path(a.output).resolve() if a.output else ROOT/'results'/time.strftime('%Y%m%d_%H%M%S')
    if output.exists() and any(output.iterdir()):
        ap.error('output must be new or empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output/'run.json').write_text(json.dumps(dict(methods=methods,H=hs,n=a.n,device=a.device,
        weights=str(weights),checkpoint_sha256=digest_file(weights) if 'main' in methods else None,
        workers=a.workers,threads_per_worker=1,test_indices=list(range(30,30+a.n)),
        alpha=1.5,N=10,p_w=.10),indent=2)+'\n',encoding='utf-8')
    jobs = [(m,h,i,a.device,str(weights)) for m in methods for h in hs for i in range(a.n)]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(a.workers,len(jobs)), mp_context=mp.get_context('spawn')) as pool:
        for future in cf.as_completed([pool.submit(evaluate_one,j) for j in jobs]):
            r = future.result()
            rows.append(r)
            write_csv(output/'per_instance.csv', sorted(rows,key=lambda x:(x['H'],x['method'],x['instance_id'])))
            print(f"[{len(rows)}/{len(jobs)}] {r['method']} H={r['H']} instance={r['instance_id']} FCT={r['mean_fct']:.4f}",flush=True)
    import numpy as np
    summary = []
    for H in hs:
        baseline = {r['instance_id']:r for r in rows if r['method']=='HEFT_Comm' and r['H']==H}
        for method in methods:
            current = sorted([r for r in rows if r['method']==method and r['H']==H],key=lambda x:x['instance_id'])
            if baseline:
                assert all(r['fixture_hash']==baseline[r['instance_id']]['fixture_hash']
                           and r['network_hash']==baseline[r['instance_id']]['network_hash'] for r in current)
            vals = np.asarray([r['mean_fct'] for r in current])
            gap = 100*(vals.mean()/np.mean([r['mean_fct'] for r in baseline.values()])-1) if baseline else None
            s = dict(method=method,dataset='Pegasus-S',H=H,n=len(vals),mean_fct=float(vals.mean()),
                     sd_ddof0=float(vals.std(ddof=0)),gap_to_HEFT_pct=gap)
            summary.append(s)
            gap_text = f'{gap:+.2f}%' if gap is not None else 'N/A'
            print(f"{method:12s} H={H:2d} n={len(vals):2d} FCT={vals.mean():.2f} ± {vals.std(ddof=0):.2f}  Gap={gap_text}")
    write_csv(output/'summary.csv',summary)
    print('Results:',output)


if __name__ == '__main__':
    main()

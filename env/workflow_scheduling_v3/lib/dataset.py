"""Pegasus-S workflows used by the main experiment."""
from pathlib import Path
import numpy as np
from env.workflow_scheduling_v3.lib import processDAG

dataset_dict = {'S': ['CyberShake_30','Montage_25','Inspiral_30','Sipht_30']}
vmVCPUs = [8,16,32,48,64,96]

def _load_one_dag(name):
    kind = name.split('_')[0]
    path = Path(__file__).resolve().parent.parent/'dax'/(name+'.xml')
    dag, total = processDAG.buildGraph(kind, str(path))
    return dag, total, kind

class dataset:
    def __init__(self, arg, vm_types):
        if arg not in dataset_dict:
            raise NotImplementedError(
                f"Unknown wf_size='{arg}'. Available: {list(dataset_dict.keys())}"
            )
        self.vmVCPU = vmVCPUs[:vm_types]  # EC2 m5
        self.meanCPU = np.mean(self.vmVCPU)
        self.wset = []
        self.wsetTotProcessTime = []
        self.wsetType = []  # 新增：记录每个 wf 的 type 标签（CyberShake / TDOA / ... ）

        for name in dataset_dict[arg]:
            dag, tot_time, type_label = _load_one_dag(name)
            for node in dag.nodes:
                dag.nodes[node]['estimatedET'] = dag.nodes[node]['taskSize'] / self.meanCPU
            self.wset.append(dag)
            self.wsetTotProcessTime.append(tot_time)
            self.wsetType.append(type_label)

        self.wsetSlowestT = []
        for app in self.wset:
            self.wsetSlowestT.append(processDAG.get_longestPath_nodeWeighted(app))

        self.wsetBeta = []
        for app in self.wset:
            self.wsetBeta.append(2)

        # self.request = np.array([1]) * 0.01  # the default is 0.01, lets test 10.0 1.0 and 0.1

        self.datacenter = [(0, 'East, USA', 0.096)]

        self.vmPrice = {2: 0.096, 4: 0.192, 8: 0.384, 16: 0.768, 32: 1.536, 48: 2.304, 64: 2.752, 96: 4.608, 128: 10.183}

"""Frozen actor and observation wrapper; no training implementation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from settings import configs
device = torch.device(configs.device)

class BatchGraph:
    def __init__(self, normalize=False):
        self.candidate_taskID = None
        self.wf_features = None
        self.wf_edges = None
        self.wf_masks = None
        self.wf_batchs = None
        self.vm_features = None
        self.vm_edges = None
        self.vm_masks = None
        self.vm_batchs = None
        self.comm_node_features = None
        self.comm_edge_index = None
        self.comm_edge_attr = None
        self.comm_batchs = None
        self.vm_id_map = None
        self.normalize = normalize

    def wrapper(
        self,
        candidate_taskID,
        wf_features,
        wf_edges,
        wf_masks,
        wf_batchs,
        vm_features,
        vm_edges,
        vm_masks,
        vm_batchs,
        comm_node_features=None,
        comm_edge_index=None,
        comm_edge_attr=None,
        vm_id_map=None,
    ):

        if self.normalize:
            wf_features = self.feature_normalize1('features', wf_features)
            vm_features = self.feature_normalize1('vm', vm_features)

        for name in ['candidate_taskID', 'wf_features', 'wf_edges', 'vm_batchs']:
            self.__dict__[name] = []
        totalTasks = 0
        for i in range(vm_features.shape[0]):
            self.candidate_taskID.append(candidate_taskID + totalTasks)
            temp_wf_fea = wf_features.copy()
            temp_wf_fea[candidate_taskID, 3:] = vm_features[i]
            self.wf_features.append(temp_wf_fea)
            self.wf_edges.append(np.column_stack([wf_edges, vm_edges[i]]) + totalTasks if len(vm_edges[i]) > 0 else (wf_edges + totalTasks))
            self.vm_batchs.append(np.full((len(wf_features),), i, dtype=np.int64))
            totalTasks += wf_features.shape[0]

        for name in ['candidate_taskID', 'wf_features', 'vm_batchs']:
            self.__dict__[name] = np.concatenate(self.__dict__[name], axis=0)
        for name in ['wf_edges']:
            self.__dict__[name] = np.concatenate(self.__dict__[name], axis=1)
        for name in ['wf_masks', 'wf_batchs']:
            self.__dict__[name] = np.tile(eval(name), vm_features.shape[0])

        self.candidate_taskID = torch.from_numpy(self.candidate_taskID).to(device)
        self.wf_features = torch.from_numpy(self.wf_features).to(device)
        self.wf_edges = torch.from_numpy(self.wf_edges.astype(np.int32)).to(device)
        self.wf_masks = torch.from_numpy(self.wf_masks).to(device)
        self.wf_batchs = torch.from_numpy(self.wf_batchs).to(device)
        self.vm_batchs = torch.from_numpy(self.vm_batchs).to(device)

        if isinstance(comm_node_features, np.ndarray) and comm_node_features.size > 0:
            self.comm_node_features = torch.from_numpy(comm_node_features.astype(np.float32)).to(device)
            self.comm_edge_index = torch.from_numpy(comm_edge_index.astype(np.int64)).to(device)
            self.comm_edge_attr = torch.from_numpy(comm_edge_attr.astype(np.float32)).to(device)
            self.comm_batchs = torch.zeros(self.comm_node_features.shape[0], dtype=torch.int64, device=device)
            self.vm_id_map = torch.from_numpy(vm_id_map.astype(np.int64)).to(device)
        else:
            self.comm_node_features = None
            self.comm_edge_index = None
            self.comm_edge_attr = None
            self.comm_batchs = None
            self.vm_id_map = None

    def feature_normalize1(self, name, data):  
        if name == 'wf':
            n = data / np.array(configs.normalize_wf)
        elif name == 'vm':
            vm_dim = data.shape[-1] if len(data.shape) > 1 else len(data[0]) if hasattr(data[0], '__len__') else 4
            if vm_dim >= 7 and hasattr(configs, 'normalize_vm_comm'):
                norm = np.array(configs.normalize_vm_comm)
                n = data / norm[:vm_dim]
            else:
                n = data / np.array(configs.normalize_vm)
        elif name == 'features':
            feat_dim = data.shape[-1] if len(data.shape) > 1 else len(data)
            if feat_dim >= 10 and hasattr(configs, 'normalize_features_comm'):
                norm = np.array(configs.normalize_features_comm)
                n = data / norm[:feat_dim]
            else:
                n = data / np.array(configs.normalize_features)
        else:
            print('Need to define the maximum value')
        return n

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_nums, activation_fn=ReLU, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        in_dim = input_dim
        for _ in range(layer_nums-1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class GATlayer(nn.Module):
    def __init__(self, in_chnl, out_chnl, dropout, concat, heads=2):   
        super().__init__()     
        self.dropout = dropout               
        self.conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat) 

    def forward(self, h_node, edge_index):     
        h_node = F.elu(self.conv(F.dropout(h_node, p=self.dropout, training=self.training).float(), edge_index))
        return h_node

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layers_gat=2, heads=2):
        super().__init__()
        self.layers_gat = layers_gat
        # self.hidden_dim = hidden_dim
        ## GAT conv layers
        self.GAT_layers = torch.nn.ModuleList()
        # init GAT layer
        if layers_gat == 1:
            # only GAT layer
            self.GAT_layers.append(GATlayer(in_dim, hidden_dim, dropout, concat=False, heads=heads))    
        else:
            # first GAT layer
            self.GAT_layers.append(GATlayer(in_dim, hidden_dim, dropout, concat=True, heads=heads))
            # following GAT layers
            for _ in range(layers_gat - 2):
                self.GAT_layers.append(GATlayer(heads * hidden_dim, hidden_dim, dropout, concat=True, heads=heads))
            # last GAT layer
            self.GAT_layers.append(GATlayer(heads * hidden_dim, hidden_dim, dropout, concat=False, heads=1))

    def forward(self, x, edge_index):

        h_node = self.GAT_layers[0](x, edge_index) 
        for layer in range(1, self.layers_gat):
            h_node = self.GAT_layers[layer](h_node, edge_index)

        return h_node

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers_gin=2):
        super(GIN, self).__init__()
        self.layer_gin = layers_gin
        self.GIN_layers = torch.nn.ModuleList()
        self.GIN_layers.append(
            GINConv(
                Sequential(Linear(in_dim, hidden_dim),
                           nn.BatchNorm1d(hidden_dim),
                           ReLU(),
                           Linear(hidden_dim, hidden_dim)),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target")
        )
        # rest gin layers
        for _ in range(layers_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim)),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target")
            )       

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        h_node = self.GIN_layers[0](x.float(), edge_index) 
        for layer in range(1, self.layer_gin):
            h_node = F.relu(h_node)
            h_node = self.GIN_layers[layer](h_node, edge_index)
        return h_node

class Actor (nn.Module):
    def __init__(self,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,                 
                 gnn_layers,
                 mlp_layers,           
                 heads=1,
                 dropout=0.0,
                 activate_fn = 'relu',
                 embedding_type = 'gat',
                 ):
        super().__init__()
        self.gnn_layers = gnn_layers
        self.mlp_layers = mlp_layers
        # self.atten_layers = atten_layers
        self.vmNum = configs.each_vm_type_num * configs.vm_types
        self.use_dual_graph = bool(getattr(configs, "use_dual_graph", 0))
        self.fusion_type = getattr(configs, "fusion_type", "gate")
        self.comm_hidden_dim = int(getattr(configs, "comm_hidden_dim", hidden_dim))
        self.comm_node_dim = int(getattr(configs, "comm_node_dim", 5))
        self.comm_edge_dim = int(getattr(configs, "comm_edge_dim", 3))
        self.comm_gnn_layers = int(getattr(configs, "comm_gnn_layers", gnn_layers))
        self.use_heuristic_prior = bool(getattr(configs, "use_heuristic_prior", 1))
        self.prior_weight = float(getattr(configs, "prior_weight", 0.35))
        self.prior_comm_coef = float(getattr(configs, "prior_comm_coef", 1.0))
        self.prior_temperature = float(getattr(configs, "prior_temperature", 2.0))
        self.runtime_prior_weight = float(self.prior_weight)
        self.runtime_temperature = 1.0
        self.runtime_top_k = 0
        self.use_vm_align_fusion = bool(getattr(configs, "use_vm_align_fusion", 1))
        self.vm_align_strict_unique = bool(getattr(configs, "vm_align_strict_unique", 1))
        self.vm_align_min_coverage = float(getattr(configs, "vm_align_min_coverage", 0.8))
        self.use_comm_edge_gating = bool(getattr(configs, "use_comm_edge_gating", 1))
        self.comm_edge_gate_scale = float(getattr(configs, "comm_edge_gate_scale", 0.2))
        self.use_comm_edge_residual = bool(getattr(configs, "use_comm_edge_residual", 1))
        self.comm_edge_residual_scale = float(getattr(configs, "comm_edge_residual_scale", 0.2))
        self.use_cross_graph_interaction = bool(getattr(configs, "use_cross_graph_interaction", 1))
        self.cross_graph_scale = float(getattr(configs, "cross_graph_scale", 0.25))
        
        ## GNN
        if embedding_type == 'gat':
            self.embedding_wf = GAT(in_dim=input_dim_wf+input_dim_vm, hidden_dim=hidden_dim, dropout=dropout, layers_gat=gnn_layers, heads=heads)
        # self.embedding_vm = GAT(in_dim=input_dim_vm, hidden_dim=hidden_dim, dropout=dropout, layers_gat=gnn_layers, heads=heads)                            
        elif embedding_type == 'gin':
            self.embedding_wf = GIN(in_dim=input_dim_wf+input_dim_vm, hidden_dim=hidden_dim, layers_gin=gnn_layers)

        # Policy
        if configs.require_mean == 1:
            in_dim = hidden_dim*2 
        else:
            in_dim = hidden_dim
        self.actor = MLP(in_dim, hidden_dim, 1, mlp_layers)
        if self.use_heuristic_prior:
            last_linear = self.actor.fc[-1]
            nn.init.uniform_(last_linear.weight, -0.01, 0.01)
            nn.init.zeros_(last_linear.bias)

        self.use_deviation_gate = bool(getattr(configs, "use_deviation_gate", 0))
        if self.use_deviation_gate:
            gate_input_dim = hidden_dim + 1
            self.deviation_gate = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            nn.init.zeros_(self.deviation_gate[2].weight)
            nn.init.constant_(self.deviation_gate[2].bias, -3.0)

        self.use_congestion_pred = bool(getattr(configs, "use_congestion_pred", 0))

        if self.use_dual_graph:
            self.embedding_comm = GAT(
                in_dim=self.comm_node_dim,
                hidden_dim=self.comm_hidden_dim,
                dropout=getattr(configs, "fusion_dropout", 0.0),
                layers_gat=self.comm_gnn_layers,
                heads=heads,
            )
            self.comm_proj = nn.Linear(self.comm_hidden_dim, hidden_dim)
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            if self.use_comm_edge_gating:
                self.comm_edge_gate = nn.Linear(self.comm_edge_dim, 1)
            if self.use_comm_edge_residual:
                self.comm_edge_residual_proj = nn.Linear(self.comm_edge_dim, hidden_dim)
            if self.use_cross_graph_interaction:
                self.cross_tc = nn.Linear(hidden_dim * 4, hidden_dim)
                self.cross_ct = nn.Linear(hidden_dim * 4, hidden_dim)
            self._warm_start_fusion_gate()
            if self.use_congestion_pred:
                self.congestion_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1)
                )

    def _warm_start_fusion_gate(self, bias_value: float = 3.0):
        """初始化融合门bias使 z=sigmoid(bias)≈0.95，初始行为接近单图模型。"""
        if hasattr(self, 'fusion_gate'):
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, bias_value)

    def _build_aligned_candidate_comm(self, comm_embed, batch_comm, vm_id_map):
        if (
            vm_id_map is None
            or batch_comm is None
            or comm_embed is None
            or comm_embed.numel() == 0
        ):
            return None
        if vm_id_map.shape[0] != comm_embed.shape[0]:
            return None

        vm_ids = vm_id_map.long()
        valid_mask = (vm_ids >= 0) & (vm_ids < self.vmNum)
        if not torch.any(valid_mask):
            return None
        batch_ids = batch_comm.long()[valid_mask]
        vm_ids = vm_ids[valid_mask]
        features = comm_embed[valid_mask]
        batch_size = int(batch_comm.max().item()) + 1
        hidden_dim = comm_embed.shape[-1]

        flat_index = batch_ids * self.vmNum + vm_ids
        aligned = torch.zeros(batch_size * self.vmNum, hidden_dim, device=comm_embed.device, dtype=comm_embed.dtype)
        counts = torch.zeros(batch_size * self.vmNum, 1, device=comm_embed.device, dtype=comm_embed.dtype)
        aligned.index_add_(0, flat_index, features)
        counts.index_add_(0, flat_index, torch.ones_like(features[:, :1]))
        if self.vm_align_strict_unique and torch.any(counts > 1.0):
            return None
        coverage = float((counts > 0).float().mean().item())
        if coverage < self.vm_align_min_coverage:
            return None
        aligned = aligned / counts.clamp_min(1.0)
        return aligned.reshape(batch_size, self.vmNum, hidden_dim)

    def _apply_edge_gating(self, comm_embed, edge_index_comm, edge_attr_comm):
        if (
            not self.use_comm_edge_gating
            or edge_index_comm is None
            or edge_attr_comm is None
            or edge_attr_comm.numel() == 0
            or comm_embed is None
            or comm_embed.numel() == 0
        ):
            return comm_embed
        if edge_index_comm.shape[1] != edge_attr_comm.shape[0]:
            return comm_embed

        edge_weight = torch.sigmoid(self.comm_edge_gate(edge_attr_comm)).squeeze(-1)
        dst = edge_index_comm[1].long()
        node_sum = torch.zeros(comm_embed.shape[0], device=comm_embed.device, dtype=comm_embed.dtype)
        node_cnt = torch.zeros(comm_embed.shape[0], device=comm_embed.device, dtype=comm_embed.dtype)
        node_sum.index_add_(0, dst, edge_weight.to(comm_embed.dtype))
        node_cnt.index_add_(0, dst, torch.ones_like(edge_weight, dtype=comm_embed.dtype))
        node_gate = (node_sum / node_cnt.clamp_min(1.0)).unsqueeze(-1)
        node_gate = 1.0 + self.comm_edge_gate_scale * (node_gate - 0.5)
        return comm_embed * node_gate

    def _apply_edge_residual(self, comm_embed, edge_index_comm, edge_attr_comm):
        if (
            not self.use_comm_edge_residual
            or edge_index_comm is None
            or edge_attr_comm is None
            or edge_attr_comm.numel() == 0
            or comm_embed is None
            or comm_embed.numel() == 0
        ):
            return comm_embed
        if edge_index_comm.shape[1] != edge_attr_comm.shape[0]:
            return comm_embed

        dst = edge_index_comm[1].long()
        edge_ctx = self.comm_edge_residual_proj(edge_attr_comm.to(comm_embed.dtype))
        node_sum = torch.zeros_like(comm_embed)
        node_cnt = torch.zeros(comm_embed.shape[0], 1, device=comm_embed.device, dtype=comm_embed.dtype)
        node_sum.index_add_(0, dst, edge_ctx)
        node_cnt.index_add_(0, dst, torch.ones_like(node_cnt[dst]))
        node_ctx = node_sum / node_cnt.clamp_min(1.0)
        return comm_embed + self.comm_edge_residual_scale * node_ctx

    def _cross_graph_interaction(self, candidate_tasks, candidate_comm):
        if (
            not self.use_cross_graph_interaction
            or candidate_tasks is None
            or candidate_comm is None
            or candidate_tasks.shape != candidate_comm.shape
        ):
            return candidate_tasks, candidate_comm
        joint = torch.cat(
            [
                candidate_tasks,
                candidate_comm,
                candidate_tasks - candidate_comm,
                candidate_tasks * candidate_comm,
            ],
            dim=-1,
        )
        gate_tc = torch.sigmoid(self.cross_tc(joint))
        gate_ct = torch.sigmoid(self.cross_ct(joint))
        enhanced_tasks = candidate_tasks + self.cross_graph_scale * gate_tc * candidate_comm
        enhanced_comm = candidate_comm + self.cross_graph_scale * gate_ct * candidate_tasks
        return enhanced_tasks, enhanced_comm

    def _fuse_with_comm(self, candidate_tasks, state_comm, edge_index_comm, edge_attr_comm, batch_comm, vm_id_map):
        if (
            not self.use_dual_graph
            or state_comm is None
            or edge_index_comm is None
            or batch_comm is None
            or state_comm.numel() == 0
        ):
            return candidate_tasks
        comm_embed = self.embedding_comm(state_comm, edge_index_comm)
        comm_embed = self.comm_proj(comm_embed)
        comm_embed = self._apply_edge_gating(comm_embed, edge_index_comm, edge_attr_comm)
        comm_embed = self._apply_edge_residual(comm_embed, edge_index_comm, edge_attr_comm)
        hidden_dim = comm_embed.shape[-1]
        candidate_comm = None
        if self.use_vm_align_fusion:
            candidate_comm = self._build_aligned_candidate_comm(comm_embed, batch_comm, vm_id_map)
        if candidate_comm is None:
            candidate_comm = comm_embed.reshape(-1, self.vmNum, hidden_dim)
        if candidate_comm.shape == candidate_tasks.shape:
            candidate_tasks, candidate_comm = self._cross_graph_interaction(candidate_tasks, candidate_comm)
            z = torch.sigmoid(self.fusion_gate(torch.cat([candidate_tasks, candidate_comm], dim=-1)))
            return z * candidate_tasks + (1 - z) * candidate_comm
        return candidate_tasks

    def _heuristic_prior_logits(self, state_wf, candidate_task_index):
        """
        Compute HEFT-equivalent EFT as prior logits.

        Features layout (after BatchGraph.wrapper inserts vm_features at col 3):
          col 3 = exec_time (ET)        — normalized by norm_vm[0]
          col 4 = rel_finish             — normalized by norm_vm[1]
          col 7 = comm_time              — normalized by norm_vm[4]

        Features are stored NORMALIZED (divided by large constants with
        different scales per column), so we must de-normalize before any
        cross-column arithmetic.

        HEFT EFT (relative) = max(vm_avail_rel, comm_time) + exec_time
        where vm_avail_rel = rel_finish - exec_time.
        """
        if not bool(getattr(self, "use_heuristic_prior", False)):
            return None
        if state_wf is None or candidate_task_index is None or state_wf.numel() == 0:
            return None
        if state_wf.shape[-1] < 5:
            return None

        candidate_raw = state_wf[candidate_task_index].reshape(-1, self.vmNum, state_wf.shape[-1])

        has_comm = candidate_raw.shape[-1] > 7

        if bool(getattr(configs, 'normalize', False)):
            if has_comm and hasattr(configs, 'normalize_vm_comm'):
                nv = configs.normalize_vm_comm
            elif hasattr(configs, 'normalize_vm'):
                nv = list(configs.normalize_vm) + [1.0] * 4
            else:
                nv = [1.0] * 8
            exec_time = candidate_raw[..., 3] * float(nv[0])
            rel_finish = candidate_raw[..., 4] * float(nv[1])
            comm_time = candidate_raw[..., 7] * float(nv[4]) if has_comm else torch.zeros_like(exec_time)
        else:
            exec_time = candidate_raw[..., 3]
            rel_finish = candidate_raw[..., 4]
            comm_time = candidate_raw[..., 7] if has_comm else torch.zeros_like(exec_time)

        vm_avail_rel = rel_finish - exec_time
        eft = torch.maximum(vm_avail_rel, comm_time) + exec_time

        min_eft = eft.min(dim=1, keepdim=True).values
        temperature = float(getattr(self, "prior_temperature", 2.0))
        prior_logits = -(eft - min_eft) / max(temperature, 0.1)

        return prior_logits.unsqueeze(-1).float()

    def set_runtime_prior_weight(self, prior_weight):
        self.runtime_prior_weight = max(0.0, float(prior_weight))

    def _apply_deviation_gate(self, graph_embed, candidate_scores, prior_logits):
        """State-dependent gating: gate modulates actor contribution based on comm state."""
        runtime_pw = float(getattr(self, "runtime_prior_weight", 0.0))
        if self.use_deviation_gate and prior_logits is not None and prior_logits.shape == candidate_scores.shape:
            _prior = prior_logits.to(graph_embed.dtype)
            gate_input = torch.cat([graph_embed, _prior], dim=-1)
            gate = self.deviation_gate(gate_input)
            return gate * candidate_scores + runtime_pw * _prior
        if prior_logits is not None and prior_logits.shape == candidate_scores.shape:
            return candidate_scores + runtime_pw * prior_logits.to(candidate_scores.dtype)
        return candidate_scores

    def forward(self,
                state_wf,      #
                state_vm,      # (18, 5)           
                edge_index_wf,        #
                edge_index_vm,        # (18, 2)           
                mask_wf,
                mask_vm,
                batch_wf,
                batch_vm,
                candidate_task_index,  ## (batch_size, n_j)  
                state_comm=None,
                edge_index_comm=None,
                edge_attr_comm=None,
                batch_comm=None,
                vm_id_map=None,
                deterministic = False,      
                # require_meanPool = True,        
                ):

        edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)
        # if configs.require_undirected == 1:
        #     edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)
            # edge_index_wf = torch.unique(edge_index_wf, dim=-1)

        wf_task_embed = self.embedding_wf(state_wf, edge_index_wf)  # (360, 32)
        hidden_dim = wf_task_embed.shape[-1] 
        candidate_tasks = wf_task_embed[candidate_task_index]  # (18, 32)
        candidate_tasks = candidate_tasks.reshape(-1, self.vmNum, hidden_dim)

        graph_embed = self._fuse_with_comm(candidate_tasks, state_comm, edge_index_comm, edge_attr_comm, batch_comm, vm_id_map)
        # if configs.require_mean == 1:
        #     average_vm_embed = global_mean_pool(wf_task_embed[mask_wf], batch_vm[mask_wf]).reshape(-1, self.vmNum, hidden_dim)  # (1, 18, 32)
        #     graph_embed = torch.cat((candidate_tasks, average_vm_embed), dim=-1)    # (1, 18, 64)
        # else:
        #     graph_embed = candidate_tasks # (1, 18, 32)

        # Calculate prob
        candidate_scores = self.actor(graph_embed)
        prior_logits = self._heuristic_prior_logits(state_wf, candidate_task_index)
        candidate_scores = self._apply_deviation_gate(graph_embed, candidate_scores, prior_logits)

        _temp = getattr(self, 'runtime_temperature', 1.0)
        _top_k = getattr(self, 'runtime_top_k', 0)
        if _temp != 1.0:
            candidate_scores = candidate_scores / _temp
        if _top_k > 0 and candidate_scores.size(1) > _top_k:
            _topv, _ = candidate_scores.topk(_top_k, dim=1)
            candidate_scores = candidate_scores.masked_fill(
                candidate_scores < _topv[:, -1:, :], float('-inf'))
        pi = F.softmax(candidate_scores, dim=1)     # (batch_Size, 18 ,1)
        dist = Categorical(probs=pi.squeeze())
        if deterministic is True:
            arr = pi.squeeze(-1)
            max_values, _ = torch.max(arr, dim=-1)
            # Keep deterministic tie-breaking indices on the same device as the
            # probability tensor.  This matters for CUDA inference; CPU-created
            # indices fail in Categorical.log_prob's gather operation.
            actions_id = torch.zeros(arr.size(0), dtype=torch.int64,
                                      device=arr.device)
            for i in range(arr.size(0)):
                max_positions = torch.nonzero(arr[i] > max_values[i] -1e-5, as_tuple=False).squeeze(1)
                random_choice = torch.randint(len(max_positions), (1,)).item()
                actions_id[i] = max_positions[random_choice]
        else:
            actions_id = dist.sample()
        log_prob = dist.log_prob(actions_id)

        return actions_id, log_prob, dist.entropy().mean()

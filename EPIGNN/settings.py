"""Fixed main-experiment inference settings; no training configuration."""
from types import SimpleNamespace

configs = SimpleNamespace(
    device='cpu',env_seed=0,env_name='WorkflowScheduling-v3',wf_num=30,wf_size='S',
    vm_types=5,each_vm_type_num=2,vmNum=10,arr_rate=5.4/3600,traf_type='CONSTANT',
    require_estimated_features=1,remove_completed=0,normalize=True,require_mean=0,
    normalize_wf=(3,39186,105873),normalize_vm=(19593,50000,48,1),
    normalize_vm_comm=(19593,50000,48,1,1000,1,1,1),
    normalize_features=(3,39186,105873,19593,50000,48,1),
    normalize_features_comm=(3,39186,105873,19593,50000,48,1,10,1,1,1),
    comm_aware=1,network_topology='hierarchical',base_bandwidth=1.,randomize_bandwidth=1,
    bandwidth_variance=.2,min_bandwidth=1.,base_latency=10.,latency_per_hop=5.,
    interference_level=.1,packet_loss_rate=.01,ew_interference_factor=.1,max_retransmit=3,
    channel_contention=1,contention_alpha=1.5,heartbeat_interval=0,
    use_dual_graph=1,comm_node_dim=7,comm_edge_dim=5,comm_hidden_dim=128,comm_gnn_layers=2,
    fusion_type='gate',fusion_dropout=0.,use_vm_align_fusion=1,vm_align_strict_unique=1,
    vm_align_min_coverage=.8,use_comm_edge_gating=1,comm_edge_gate_scale=.2,
    use_comm_edge_residual=1,comm_edge_residual_scale=.2,use_cross_graph_interaction=1,
    cross_graph_scale=.25,use_heuristic_prior=1,prior_weight=.1,prior_temperature=2.,
    prior_comm_coef=1.,use_deviation_gate=0,use_congestion_pred=1,
)

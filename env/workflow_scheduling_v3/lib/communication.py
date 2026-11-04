"""
UAV集群通信模型 

该模块实现了无人机集群的通信建模，包括：
- 多种网络拓扑支持 (mesh, star, ring, hierarchical)
- 带宽和延迟计算
- 干扰模拟
- 丢包和重传机制
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class CommunicationModel:
    """UAV集群通信模型 - 支持干扰"""
    
    def __init__(self, args):
        """
        初始化通信模型
        
        Args:
            args: 配置参数，包含通信相关设置
        """
        # 基础参数
        self.base_bandwidth = getattr(args, 'base_bandwidth', 100.0)  # Mbps
        self.bandwidth_variance = getattr(args, 'bandwidth_variance', 0.2)
        self.min_bandwidth = getattr(args, 'min_bandwidth', 10.0)
        self.base_latency = getattr(args, 'base_latency', 5.0)  # ms
        self.latency_per_hop = getattr(args, 'latency_per_hop', 2.0)
        self.packet_loss_rate = getattr(args, 'packet_loss_rate', 0.01)
        self.ew_factor = getattr(args, 'ew_interference_factor', 0.1)
        self.max_retransmit = getattr(args, 'max_retransmit', 3)
        
        # 带宽矩阵 [node_i, node_j] -> bandwidth (Mbps)
        self.bandwidth_matrix: Optional[np.ndarray] = None
        # 跳数矩阵 [node_i, node_j] -> hops
        self.hop_matrix: Optional[np.ndarray] = None
        # 当前干扰等级 (0-1)
        self.interference_level = getattr(args, 'interference_level', 0.0)
        # 节点数量
        self.num_nodes = 0
        
        # ===== 逐链路拥塞模型 =====
        self.channel_contention = bool(getattr(args, 'channel_contention', 0))
        self.contention_alpha = getattr(args, 'contention_alpha', 1.0)
        self._link_transfers: Dict[Tuple[int, int], List[float]] = {}
        self._active_transfers: List[float] = []
        
    def initialize_network(self, num_nodes: int, topology: str = 'mesh'):
        """
        初始化网络拓扑
        
        Args:
            num_nodes: 节点数量
            topology: 拓扑类型 ('mesh', 'star', 'ring', 'hierarchical')
        """
        self.num_nodes = num_nodes
        self._topology_name = str(topology)
        # Ring bandwidth already contains the 1/hops path penalty.  Apply the
        # physical-link floor at that same scale, instead of lifting every
        # multi-hop path back to a one-hop bandwidth.  Historical hierarchical
        # cost semantics stay unchanged.
        self._route_bandwidth_floor = (topology == 'ring')
        
        if topology == 'mesh':
            # 全连接网格：所有节点直连
            self.bandwidth_matrix = np.full((num_nodes, num_nodes), 
                                            self.base_bandwidth)
            self.hop_matrix = np.ones((num_nodes, num_nodes), dtype=int)
            
        elif topology == 'star':
            # 星型拓扑：通过中心节点(0)通信
            self.bandwidth_matrix = np.zeros((num_nodes, num_nodes))
            self.hop_matrix = np.full((num_nodes, num_nodes), 2, dtype=int)
            for i in range(num_nodes):
                self.bandwidth_matrix[0, i] = self.base_bandwidth
                self.bandwidth_matrix[i, 0] = self.base_bandwidth
                self.hop_matrix[0, i] = 1
                self.hop_matrix[i, 0] = 1
            # 非中心节点之间需要经过中心
            for i in range(1, num_nodes):
                for j in range(1, num_nodes):
                    if i != j:
                        self.bandwidth_matrix[i, j] = self.base_bandwidth * 0.8
                        
        elif topology == 'ring':
            # 环形拓扑
            self.bandwidth_matrix = np.zeros((num_nodes, num_nodes))
            self.hop_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            for i in range(num_nodes):
                next_i = (i + 1) % num_nodes
                self.bandwidth_matrix[i, next_i] = self.base_bandwidth
                self.bandwidth_matrix[next_i, i] = self.base_bandwidth
                # 计算跳数（环形最短路径）
                for j in range(num_nodes):
                    self.hop_matrix[i, j] = min(abs(i-j), num_nodes - abs(i-j))
            # 填充非直连节点的带宽（通过多跳）
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if self.bandwidth_matrix[i, j] == 0 and i != j:
                        hops = self.hop_matrix[i, j]
                        self.bandwidth_matrix[i, j] = self.base_bandwidth / hops
                        
        elif topology == 'hierarchical':
            # 层级拓扑：适合无人机编队 (leader-follower)
            self._init_hierarchical_topology(num_nodes)
            
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # 对角线设为无穷大（本地不需要通信）
        np.fill_diagonal(self.bandwidth_matrix, float('inf'))
        np.fill_diagonal(self.hop_matrix, 0)
        # Retain the physical one-hop graph separately from the effective
        # all-pairs matrices.  This is used only by the robustness scenarios;
        # the historical communication model continues to use the matrices
        # above unchanged.
        self._physical_adjacency = self._physical_graph_for_topology(num_nodes, topology)
        self._physical_bandwidth = np.where(
            self._physical_adjacency > 0, self.bandwidth_matrix, 0.0).astype(float)

    def _physical_graph_for_topology(self, n: int, topology: str) -> np.ndarray:
        """Return an undirected physical one-hop graph for fault experiments."""
        a = np.zeros((n, n), dtype=float)
        if topology == 'mesh':
            a[:] = 1.0
            np.fill_diagonal(a, 0.0)
        elif topology == 'star' or topology == 'hierarchical':
            for i in range(1, n):
                a[0, i] = a[i, 0] = 1.0
        elif topology == 'ring':
            for i in range(n):
                j = (i + 1) % n
                a[i, j] = a[j, i] = 1.0
        else:
            raise ValueError(f"Unknown topology: {topology}")
        return a

    def _recompute_effective_from_physical(self):
        """Recompute all-pairs hop/bottleneck bandwidth after a link change."""
        n = int(self.num_nodes)
        adj = np.asarray(self._physical_adjacency, dtype=float)
        edge_bw = np.asarray(self._physical_bandwidth, dtype=float)
        hops = np.full((n, n), 100, dtype=int)
        bw = np.zeros((n, n), dtype=float)
        for s in range(n):
            hops[s, s] = 0
            bw[s, s] = np.inf
            # Shortest-hop routing, with max-min bandwidth tie breaking.
            best = np.zeros(n, dtype=float); best[s] = np.inf
            dist = np.full(n, 100, dtype=int); dist[s] = 0
            frontier = [s]
            for _ in range(n):
                nxt = []
                for u in frontier:
                    for v in np.flatnonzero(adj[u] > 0):
                        cand_d = dist[u] + 1
                        cand_bw = min(best[u], float(edge_bw[u, v]))
                        if cand_d < dist[v] or (cand_d == dist[v] and cand_bw > best[v]):
                            dist[v] = cand_d; best[v] = cand_bw; nxt.append(int(v))
                frontier = nxt
                if not frontier: break
            for d in range(n):
                if d != s and dist[d] < 100:
                    hops[s, d] = dist[d]
                    bw[s, d] = best[d] / max(1, dist[d])
        self.hop_matrix = hops
        self.bandwidth_matrix = bw

        
    def _init_hierarchical_topology(self, num_nodes: int):
        """
        层级拓扑初始化 - 一个领导者连接多个跟随者
        
        结构：节点0是领导者，其他节点是跟随者
        领导者与所有跟随者直连，跟随者之间需要经过领导者
        
        当 randomize_bandwidth=True 时，每个 follower 到 leader 的带宽
        从 [0.3, 2.0] × base_bandwidth 均匀采样，模拟 UAV 编队中
        不同无人机到指挥节点的距离差异。
        """
        self.bandwidth_matrix = np.zeros((num_nodes, num_nodes))
        self.hop_matrix = np.full((num_nodes, num_nodes), 2, dtype=int)
        
        randomize = bool(getattr(self, '_randomize_bw', False))
        
        # 节点0是领导者，连接所有跟随者
        for i in range(1, num_nodes):
            if randomize:
                factor = np.random.uniform(0.3, 2.0)
            else:
                factor = 1.5
            bw = self.base_bandwidth * factor
            self.bandwidth_matrix[0, i] = bw
            self.bandwidth_matrix[i, 0] = bw
            self.hop_matrix[0, i] = 1
            self.hop_matrix[i, 0] = 1
            
        # 跟随者之间通过领导者通信
        for i in range(1, num_nodes):
            for j in range(i+1, num_nodes):
                self.bandwidth_matrix[i, j] = min(
                    self.bandwidth_matrix[i, 0],
                    self.bandwidth_matrix[0, j]
                ) * 0.8
                self.bandwidth_matrix[j, i] = self.bandwidth_matrix[i, j]
                
        
    def set_interference_level(self, level: float):
        """
        设置干扰等级
        
        Args:
            level: 干扰等级 (0-1)，0表示无干扰，1表示最大干扰
        """
        self.interference_level = np.clip(level, 0.0, 1.0)
        
    def get_effective_bandwidth(self, src: int, dst: int) -> float:
        """
        获取有效带宽（考虑干扰）
        
        Args:
            src: 源节点索引
            dst: 目标节点索引
            
        Returns:
            有效带宽 (Mbps)
        """
        if src == dst:
            return float('inf')  # 本地无通信开销
            
        base_bw = self.bandwidth_matrix[src, dst]
        if base_bw == 0:
            return 0  # 无直连
            
        # 干扰影响：增加随机波动
        noise = np.random.normal(0, self.bandwidth_variance * self.interference_level)
        # 干扰降低带宽
        interference_reduction = self.interference_level * self.ew_factor * base_bw
        
        effective_bw = base_bw * (1 - abs(noise)) - interference_reduction
        return max(effective_bw, self.get_path_bandwidth_floor(src, dst))

    def get_path_bandwidth_floor(self, src: int, dst: int, hop_matrix=None) -> float:
        """Translate a physical ring link floor into an end-to-end path floor.

        A snapshot hop_matrix must be supplied for stale network observations.
        """
        if getattr(self, '_route_bandwidth_floor', False):
            hops = self.hop_matrix if hop_matrix is None else hop_matrix
            return float(self.min_bandwidth) / max(1, int(hops[src, dst]))
        return float(self.min_bandwidth)
    
    def calculate_transfer_time(self, src: int, dst: int, 
                                 data_size_mb: float) -> Tuple[float, float]:
        """
        计算数据传输时间
        
        Args:
            src: 源节点索引
            dst: 目标节点索引
            data_size_mb: 数据大小 (MB)
            
        Returns:
            (transfer_time_seconds, success_probability)
        """
        if src == dst or data_size_mb <= 0:
            return 0.0, 1.0
            
        bandwidth = self.get_effective_bandwidth(src, dst)
        if bandwidth <= 0:
            return float('inf'), 0.0
            
        # 传输时间 = 数据量(Mb) / 带宽(Mbps) = seconds
        # data_size_mb * 8 = Megabits
        transfer_time = (data_size_mb * 8) / bandwidth  # seconds
        
        # 基础延迟
        hops = self.hop_matrix[src, dst]
        latency = (self.base_latency + hops * self.latency_per_hop) / 1000  # ms -> seconds
        
        # 干扰增加延迟
        interference_latency = self.interference_level * 0.05  # 最多增加50ms
        
        total_time = transfer_time + latency + interference_latency
        
        # 成功概率（考虑丢包）
        effective_loss_rate = self.packet_loss_rate * (1 + self.interference_level * 5)
        success_prob = (1 - effective_loss_rate) ** hops
        
        # 考虑重传的期望时间
        if success_prob > 0:
            expected_attempts = min(1 / success_prob, self.max_retransmit)
        else:
            expected_attempts = self.max_retransmit
        expected_time = total_time * expected_attempts
        
        return expected_time, success_prob
    
    def get_communication_cost(self, src: int, dst: int, 
                                data_size_mb: float) -> float:
        """
        获取通信代价（用于HEFT和RL状态）
        
        这是调度决策使用的主要接口
        
        Args:
            src: 源节点索引
            dst: 目标节点索引
            data_size_mb: 数据大小 (MB)
            
        Returns:
            通信代价（秒）
        """
        if data_size_mb <= 0 or src == dst:
            return 0.0
            
        transfer_time, success_prob = self.calculate_transfer_time(src, dst, data_size_mb)
        
        # 如果成功率太低，增加惩罚
        if success_prob < 0.9:
            reliability_penalty = (1 - success_prob) * 1.0  # 最多增加1秒惩罚
        else:
            reliability_penalty = 0
        
        return transfer_time + reliability_penalty
    
    # ================================================================
    #  共享信道拥塞追踪
    # ================================================================

    def reset_transfers(self):
        """环境 reset 时清空所有活跃传输"""
        self._link_transfers.clear()
        self._active_transfers = []

    def register_transfer(self, end_time: float, src: int = -1, dst: int = -1):
        """注册一个数据传输。支持逐链路 (src, dst) 和全局兼容模式。"""
        self._active_transfers.append(end_time)
        if src >= 0 and dst >= 0 and src != dst:
            key = (src, dst)
            if key not in self._link_transfers:
                self._link_transfers[key] = []
            self._link_transfers[key].append(end_time)

    def _cleanup_link_transfers(self, current_time: float):
        """清理所有链路上已完成的传输"""
        empty_keys = []
        for key, times in self._link_transfers.items():
            self._link_transfers[key] = [t for t in times if t > current_time]
            if not self._link_transfers[key]:
                empty_keys.append(key)
        for key in empty_keys:
            del self._link_transfers[key]

    def get_link_transfer_count(self, src: int, dst: int,
                                current_time: float) -> int:
        """返回指定链路 (src→dst) 上的活跃传输数"""
        key = (src, dst)
        if key not in self._link_transfers:
            return 0
        self._link_transfers[key] = [
            t for t in self._link_transfers[key] if t > current_time
        ]
        if not self._link_transfers[key]:
            del self._link_transfers[key]
            return 0
        return len(self._link_transfers[key])

    def get_active_transfer_count(self, current_time: float) -> int:
        """返回全局活跃传输总数（向后兼容）"""
        self._active_transfers = [t for t in self._active_transfers if t > current_time]
        return len(self._active_transfers)

    def get_congestion_level(self, current_time: float) -> float:
        """全局归一化拥塞度 [0, 1]"""
        active = self.get_active_transfer_count(current_time)
        return min(1.0, active / max(1, self.num_nodes))

    def get_node_radio_load(self, node: int, current_time: float) -> int:
        """返回某节点无线电台的总负载（所有参与的传输，包括发送和接收）"""
        self._cleanup_link_transfers(current_time)
        load = 0
        for (s, d), times in self._link_transfers.items():
            if s == node or d == node:
                load += len(times)
        return load

    def get_node_load_vector(self, current_time: float) -> np.ndarray:
        """返回 [N] 向量，每个元素是该节点的无线电台总负载。"""
        self._cleanup_link_transfers(current_time)
        n = int(self.num_nodes)
        vec = np.zeros(n, dtype=np.float32)
        for (s, d), times in self._link_transfers.items():
            cnt = len(times)
            if s < n:
                vec[s] += cnt
            if d < n:
                vec[d] += cnt
        return vec

    def get_link_congestion_matrix(self, current_time: float) -> np.ndarray:
        """返回 [N, N] 矩阵，每个元素是该链路的活跃传输数。"""
        self._cleanup_link_transfers(current_time)
        n = int(self.num_nodes)
        mat = np.zeros((n, n), dtype=np.float32)
        for (s, d), times in self._link_transfers.items():
            if s < n and d < n:
                mat[s, d] = len(times)
        return mat

    def congestion_multiplier(self, load):
        # Fixed alternative; no parameter fitting or checkpoint selection.
        load = max(0.0, float(load))
        return (1.0 + load) ** self.contention_alpha

    def get_communication_cost_congested(self, src: int, dst: int,
                                         data_size_mb: float,
                                         current_time: float) -> float:
        """
        UAV 逐节点无线电台拥塞模型。
        瓶颈 = 发送方和接收方中更繁忙的那个节点的无线电台负载。
        有效带宽 = 基础带宽 / (1 + max(src_load, dst_load))
        """
        if data_size_mb <= 0 or src == dst:
            return 0.0

        if not self.channel_contention:
            return self.get_communication_cost(src, dst, data_size_mb)

        src_load = self.get_node_radio_load(src, current_time)
        dst_load = self.get_node_radio_load(dst, current_time)
        congestion_divisor = self.congestion_multiplier(max(src_load, dst_load))

        base_bw = self.bandwidth_matrix[src, dst]
        if base_bw == 0 or not np.isfinite(base_bw):
            return self.get_communication_cost(src, dst, data_size_mb)

        noise = np.random.normal(0, self.bandwidth_variance * self.interference_level)
        interference_reduction = self.interference_level * self.ew_factor * base_bw
        effective_bw = base_bw * (1 - abs(noise)) - interference_reduction
        effective_bw = max(effective_bw, self.get_path_bandwidth_floor(src, dst))
        effective_bw = effective_bw / congestion_divisor

        transfer_time = (data_size_mb * 8) / max(effective_bw, 0.01)

        hops = int(self.hop_matrix[src, dst])
        latency = (self.base_latency + hops * self.latency_per_hop) / 1000
        interference_latency = self.interference_level * 0.05

        total_time = transfer_time + latency + interference_latency

        effective_loss_rate = self.packet_loss_rate * (1 + self.interference_level * 5)
        success_prob = (1 - effective_loss_rate) ** hops
        if success_prob > 0:
            expected_attempts = min(1 / success_prob, self.max_retransmit)
        else:
            expected_attempts = self.max_retransmit

        cost = total_time * expected_attempts
        if success_prob < 0.9:
            cost += (1 - success_prob) * 1.0
        return cost

    def get_network_state(self) -> Dict:
        """
        获取网络状态信息（可用于RL状态特征）
        
        Returns:
            包含网络状态的字典
        """
        return {
            'num_nodes': self.num_nodes,
            'interference_level': self.interference_level,
            'avg_bandwidth': np.mean(self.bandwidth_matrix[self.bandwidth_matrix < float('inf')]),
            'avg_hops': np.mean(self.hop_matrix[self.hop_matrix > 0]) if np.any(self.hop_matrix > 0) else 0,
        }
    

    def export_graph_tensors(self, current_time: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        导出通信图张量 — UAV 逐节点无线电台拥塞模型。

        Returns:
            node_features: [N, 7] => [self_bw, avg_bw, avg_hops, interference, degree,
                                       radio_load_norm, radio_bw_factor]
            edge_index: [2, E]
            edge_attr: [E, 5] => [bw_norm, hop_norm, interference, effective_bw_ratio,
                                   bottleneck_load_norm]
        """
        NODE_DIM = 7
        EDGE_DIM = 5
        n = int(self.num_nodes)
        if n <= 0:
            return (
                np.zeros((0, NODE_DIM), dtype=np.float32),
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, EDGE_DIM), dtype=np.float32),
            )

        bw = self.bandwidth_matrix if self.bandwidth_matrix is not None else np.zeros((n, n), dtype=np.float32)
        hops_mat = self.hop_matrix if self.hop_matrix is not None else np.zeros((n, n), dtype=np.int64)
        bw_norm_base = max(float(self.base_bandwidth), 1e-6)

        has_congestion = (
            self.channel_contention
            and current_time is not None
            and current_time >= 0
        )
        if has_congestion:
            node_load = self.get_node_load_vector(current_time)
        else:
            node_load = np.zeros(n, dtype=np.float32)

        node_features = np.zeros((n, NODE_DIM), dtype=np.float32)
        edges_src: List[int] = []
        edges_dst: List[int] = []
        edge_attrs: List[List[float]] = []

        for i in range(n):
            valid_bw = []
            valid_hops = []
            degree = 0
            for j in range(n):
                if i == j:
                    continue
                bw_ij = float(bw[i, j]) if np.isfinite(bw[i, j]) else 0.0
                hop_ij = int(hops_mat[i, j]) if np.isfinite(hops_mat[i, j]) else 100
                if bw_ij > 0:
                    degree += 1
                    valid_bw.append(bw_ij)
                    valid_hops.append(hop_ij)
                    edges_src.append(i)
                    edges_dst.append(j)

                    bottleneck = max(node_load[i], node_load[j])
                    divisor = 1 + bottleneck
                    eff_ratio = (bw_ij / divisor) / bw_norm_base
                    bottleneck_norm = min(bottleneck / max(1, n), 1.0)

                    edge_attrs.append([
                        bw_ij / bw_norm_base,
                        min(float(hop_ij) / 10.0, 1.0),
                        float(self.interference_level),
                        eff_ratio,
                        bottleneck_norm,
                    ])

            avg_bw = (np.mean(valid_bw) / bw_norm_base) if valid_bw else 0.0
            avg_hops = min((np.mean(valid_hops) / 10.0), 1.0) if valid_hops else 0.0
            self_bw = 1.0
            degree_norm = degree / max(1, n - 1)
            load_i = float(node_load[i])
            radio_load_norm = min(load_i / max(1, n), 1.0)
            radio_bw_factor = 1.0 / (1 + load_i)
            node_features[i] = np.array(
                [self_bw, avg_bw, avg_hops, float(self.interference_level), degree_norm,
                 radio_load_norm, radio_bw_factor],
                dtype=np.float32,
            )

        edge_index = (
            np.array([edges_src, edges_dst], dtype=np.int64)
            if len(edges_src) > 0
            else np.zeros((2, 0), dtype=np.int64)
        )
        edge_attr = (
            np.array(edge_attrs, dtype=np.float32)
            if len(edge_attrs) > 0
            else np.zeros((0, EDGE_DIM), dtype=np.float32)
        )

        return node_features, edge_index, edge_attr


            
    def _compute_hop_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Floyd-Warshall算法计算最短跳数
        
        Args:
            adjacency: 邻接矩阵
            
        Returns:
            跳数矩阵
        """
        n = adjacency.shape[0]
        hops = np.where(adjacency > 0, 1, float('inf'))
        np.fill_diagonal(hops, 0)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if hops[i, k] + hops[k, j] < hops[i, j]:
                        hops[i, j] = hops[i, k] + hops[k, j]
                        
        # 将inf替换为一个大数（表示不可达）
        hops = np.where(np.isinf(hops), 100, hops)
        return hops.astype(int)
    


# 便捷函数



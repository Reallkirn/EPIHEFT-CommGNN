import xml.etree.ElementTree as ET
import networkx as nx

def buildGraph(type, filename, add_data_size=True):
    """
    解析DAG并添加边上的数据传输量
    
    Args:
        type: 工作流类型名称
        filename: XML文件路径
        add_data_size: 是否在边上添加数据传输量（用于通信感知调度）
        
    Returns:
        dag: 任务DAG (networkx.DiGraph)
        tot_processTime: 总处理时间
    """
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    
    with open(filename, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        xml_file.close()
    root = tree.getroot()
    
    # 第一遍：解析所有节点
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            size = 0
            output_size = 0  # 输出文件大小
            for p in child:
                file_size = int(p.attrib['size'])
                size += file_size
                # 检查是否是输出文件
                if p.attrib.get('link') == 'output':
                    output_size += file_size
            
            # 如果没有明确的输出文件，估计为总大小的10%
            if output_size == 0:
                output_size = size * 0.1
                
            dag.add_node(
                int(child.attrib['id'][2:]), 
                taskSize=float(child.attrib['runtime']) * 16, 
                size=size,
                output_size=output_size  # 新增：输出数据大小
            )
            tot_processTime += float(child.attrib['runtime']) * 16
    
    # 第二遍：解析边并添加数据量
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                
                if add_data_size and parent in dag.nodes:
                    # 边上数据量 = 父节点输出数据大小
                    parent_output = dag.nodes[parent].get('output_size', 
                                    dag.nodes[parent].get('size', 1000) * 0.1)
                    # 转换为MB (假设原始单位是bytes)
                    data_size_mb = parent_output / (1024 * 1024)
                    # 确保最小数据量
                    data_size_mb = max(data_size_mb, 0.01)  # 最小0.01MB
                else:
                    data_size_mb = 0.0
                    
                dag.add_edge(parent, kid, data_size=data_size_mb)
                
    return dag, tot_processTime


def get_longestPath_nodeWeighted(G):
    entrance = [n for n, d in G.in_degree() if d == 0]
    exit = [n for n, d in G.out_degree() if d == 0]
    cost = 0
    for root in entrance:
        for leaf in exit:
            for path in nx.all_simple_paths(G, source=root, target=leaf):  
                temp = sum(G.nodes[node]['taskSize'] for node in path)
                if temp > cost:
                    cost = temp
    return cost

def get_shortest_weighted_path_sum(G):
    entrance = [n for n, d in G.in_degree() if d == 0]
    exit = [n for n, d in G.out_degree() if d == 0]
    cost = 0
    for root in entrance:
        for leaf in exit:
            try:
                shortest_path = nx.shortest_path(G, source=root, target=leaf, weight='taskSize', method='dijkstra')
                path_sum = sum(G.nodes[node]['taskSize'] for node in shortest_path)
                if path_sum > cost:
                    cost = path_sum
            except nx.NetworkXNoPath:
                cost = float('inf')
    return cost


def get_shortest_paths_to_end(G, start):
    """
    Calculate the shortest paths from any node in G to the target node.
    
    :param G: A directed graph (networkx DiGraph)
    :param start: The start node
    :return: A dictionary with nodes as keys and shortest path lengths to the target as values
    """
    shortest_paths = {}
    for node in G.nodes:
        try:
            # Using Dijkstra's algorithm to find the shortest path length from node to target
            path_length = nx.shortest_path_length(G, source=start, target=node, weight='taskSize', method='dijkstra')
            shortest_paths[node] = path_length
        except nx.NetworkXNoPath:
            # If there is no path from the node to the target, set the path length to infinity
            shortest_paths[node] = float('inf')
    return shortest_paths

def compute_initial_earliest_finish_times(G, generateTime):
    # Topological sort
    topo_order = list(nx.topological_sort(G))
    
    # Initialize the earliest end time of all nodes to 0
    earliest_finish_times = {node: generateTime for node in G.nodes}
    
    # Traverse the topologically sorted nodes
    for node in topo_order:
        # The start time of the current node is the maximum value of the earliest end time of all predecessor nodes
        if G.in_edges(node):
            max_predecessor_finish_time = max(earliest_finish_times[predecessor] for predecessor in G.predecessors(node))
        else:
            max_predecessor_finish_time = generateTime
        
        # The earliest end time of the current node = the start time of the current node + the processing time of the current node
        earliest_finish_times[node] = max_predecessor_finish_time + G.nodes[node]['estimatedET']
    
    return earliest_finish_times

def update_earliest_finish_times(G, node, actual_finish_time, earliest_finish_times):
    # Update the earliest end time of the current node to the actual completion time
    if abs(earliest_finish_times[node] - actual_finish_time) > 1e-5:
        earliest_finish_times[node] = actual_finish_time
        
        # Use a queue to perform a breadth-first search to update the earliest end time of all successor nodes
        queue = [node]
        
        while queue:
            current_node = queue.pop(0)
            
            for successor in G.successors(current_node):
                # The start time of the current node is the maximum value of the earliest end time of all predecessor nodes
                max_predecessor_finish_time = max(earliest_finish_times[predecessor] for predecessor in G.predecessors(successor))
                
                # The earliest end time of the current node = the start time of the current node + the processing time of the current node
                new_finish_time = max_predecessor_finish_time + G.nodes[successor]['estimatedET']
                
                if new_finish_time > earliest_finish_times[successor]:
                    earliest_finish_times[successor] = new_finish_time
                    queue.append(successor)
    
    return earliest_finish_times

def update_node_weight_and_compute_finish_times(G, generateTime, node, new_weight, actual_finish_times):
    G.nodes[node]['estimatedET'] = new_weight  # replace weights
    earliest_finish_times = compute_initial_earliest_finish_times(G, generateTime)
    for idx, actual_finish_time in actual_finish_times.items(): 
        earliest_finish_times = update_earliest_finish_times(G, idx, actual_finish_time, earliest_finish_times)

    return earliest_finish_times 

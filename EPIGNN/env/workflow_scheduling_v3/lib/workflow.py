import copy
import networkx as nx
import numpy as np

class Workflow:
    def __init__(self, generateTime1, app, appID, originDC, time, totalT, index): #, ratio):
        self.generateTime = generateTime1
        self.origin = originDC # comes from which user
        self.app = app  # e.g., w1 = [n1, n2, n3, n4, n5, n6],  n1 = [[], [2], 's1'], n2 = [[1], [3], 's14']
        self.tempDAG = copy.deepcopy(app)
        self.taskNum = app.number_of_nodes()
        self.appID = appID ## app is workflow；workflow type
        self.appArivalIndex = index # used to differentiate different workflows. A workflow includes a number of different tasks 
        self.readyTime = {} # ready time
        self.enqueueTime = {}  # {nodeID: enqueueTime for the service}  # StartTime
        self.dequeueTime = {}  # {nodeID: dequeueTime for each service} to ensure all parallel services are completed before the aggregated service starts
        self.pendingIndexOnDC ={}
        self.processDC = {}  # {nodeID: dc_ind} ensure parallel services are aggregated to the same destination service
        self.executeTime = {}
        self.waitingTime = {}
        self.queuingTask = []  # tasks that are currently in queue. Mostly for parallel services
        self.maxProcessTime = time  # the max processing time for the app if parallel processing is enabled and no propagation latency
        self.totalProcessTime = totalT
        self.remainWorkload = totalT
        self.mask = np.full(shape=app.number_of_nodes(), fill_value=1, dtype=bool)
        self.actual_finish_time = {} #{node: generateTime1 for node in app.nodes}
        # self.dueTimeCoef = ratio
        # self.deadlineTime = self.maxProcessTime * self.dueTimeCoef + self.generateTime

    def __lt__(self, other):  
        return self.generateTime < other.generateTime

    def get_completeTaskNum(self):
        return len(self.dequeueTime)

    def get_maxProcessTime(self):
        return self.maxProcessTime

    # return True if tasks in s_list are completed, otherwise False
    def is_completeTaskSet(self, s_list):
        if set(s_list).issubset(set(self.dequeueTime.keys())):
            return True
        else:
            return False

    # return the set of tasks in s_list that are completed
    def completeTaskSet(self, s_list):
        return set(s_list).intersection(set(self.dequeueTime.keys()))
    

    def update_pendingIndexVM(self, task, vmid, pendingTndex, speed, util):
        self.pendingIndexOnDC[task] = [vmid,  pendingTndex, speed, util]

    # time: enqueue time for "service" which is allocated in "dc"
    def update_enqueueTime(self, time, task, vmID):
        if task not in self.enqueueTime:
            self.readyTime[task] = time
            self.enqueueTime[task] = time
            self.processDC[task] = vmID  # guarantee results from all parallel services aggregating at the same destination

        elif time > self.enqueueTime[task]:
            self.enqueueTime[task] = time   # start time 

    def update_dequeueTime(self, time, task):
        self.dequeueTime[task] = time
        if task is not None:
            self.mask[task] = False
            self.remainWorkload -= self.get_taskSize(task)
            self.tempDAG.remove_node(task)
            self.waitingTime[task] = self.enqueueTime[task] - self.readyTime[task]
            self.actual_finish_time[task] = time 

    def update_executeTime(self, time, task):
        self.executeTime[task] = time

    def update_waitingTime(self, time, task):
        self.waitingTime[task] = time

    def get_waitingTime(self, task):
        return self.waitingTime[task] if task is not None else 0

    def get_executeTime(self, task):
        return self.executeTime[task]
  
    def add_queuingTask(self, task):
        self.queuingTask.append(task)

    def remove_queuingTask(self, task):
        self.queuingTask.remove(task)

    def get_taskSize(self, task):
        return self.app.nodes[task]['taskSize'] if task is not None else 0

    def get_taskDC(self, task):
        if task in self.processDC:
            return self.processDC[task]
        else:
            return None

    def get_generateTime(self):
        return self.generateTime

    def get_enqueueTime(self, task):
        return self.enqueueTime[task]

    def get_readyTime(self, task):
        return self.readyTime[task]

    def get_originDC(self):
        return self.origin

    def get_wfID(self):
        return self.appArivalIndex
    
    def get_wfType(self):
        return self.appID    

    def get_taskProcessTime(self, task):
        return self.app.nodes[task]['taskSize']  # networkx
        # return self.app.vs[task]['taskSize']  # igraph

    # return the next task list given the current task ID
    def get_allnextTask(self, task):
        if task is None:
            root = [n for n, d in self.app.in_degree() if d == 0]  # networkx
            return root
        else:
            return list(self.app.successors(task))

    def get_NumofSuccessors(self, task):          
        node_succ = self.get_allnextTask(task)
        return len(node_succ)

    def get_Deadline(self):              
        return self.deadlineTime 
 
    # return the parent tasks given the current task
    def get_allpreviousTask(self, task):
        if task is None:
            return []
        else:
            return list(self.app.predecessors(task))

    def get_totNumofTask(self):
        return self.app.number_of_nodes()  # networkx
        # return self.app.vcount()    # igraph

    def get_allTask(self):
        return list(self.app.nodes)  # networkx
        # return self.app.vs.indices   # igraph

    def get_path2end(self, start):
        G = self.app
        exit = [n for n, d in G.out_degree() if d == 0]
        cost = 1e10
        for leaf in exit:
            for path in nx.all_simple_paths(G, source=start, target=leaf):   
                temp = sum(G.nodes[node]['taskSize'] for node in path)
                if temp < cost:
                    cost = temp
        return cost-G.nodes[start]['taskSize']
    
    def get_allsucc2end(self, start):       # task后所有子代的sum of task size
        G = self.app
        exit = [n for n, d in G.out_degree() if d == 0]     # 最后的点
        nodes_in_paths = set()
        for leaf in exit:
            for path in nx.all_simple_paths(G, source=start, target=leaf):   
                nodes_in_paths.update(path)
        total = sum(G.nodes[node]['taskSize'] for node in nodes_in_paths)

        return total-G.nodes[start]['taskSize']
    
    # ============ 通信相关方法 ============
    def get_data_size(self, parent_task: int, child_task: int) -> float:
        """
        获取两个任务之间的数据传输量
        
        Args:
            parent_task: 父任务ID
            child_task: 子任务ID
            
        Returns:
            数据大小 (MB)
        """
        if self.app.has_edge(parent_task, child_task):
            return self.app.edges[parent_task, child_task].get('data_size', 0.0)
        return 0.0
    
    def get_total_input_data(self, task: int) -> float:
        """
        获取任务的总输入数据量（来自所有前驱任务）
        
        Args:
            task: 任务ID
            
        Returns:
            总输入数据大小 (MB)
        """
        total = 0.0
        for parent in self.get_allpreviousTask(task):
            total += self.get_data_size(parent, task)
        return total
    
    def get_output_data_size(self, task: int) -> float:
        """
        获取任务的输出数据大小
        
        Args:
            task: 任务ID
            
        Returns:
            输出数据大小 (MB)
        """
        if task is None:
            return 0.0
        output_bytes = self.app.nodes[task].get('output_size', 
                       self.app.nodes[task].get('size', 1000) * 0.1)
        return output_bytes / (1024 * 1024)  # 转换为MB    
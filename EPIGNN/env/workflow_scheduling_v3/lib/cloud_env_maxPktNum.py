import numpy as np
import pandas as pd
import csv
import math
import hashlib
import os, sys, inspect, random, copy
try:
    import gymnasium as gym
except ImportError:  # fallback for older environments
    import gym
from settings import configs
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.stats import Stats
from env.workflow_scheduling_v3.lib.poissonSampling import one_sample_poisson
from env.workflow_scheduling_v3.lib.vm import VM
from env.workflow_scheduling_v3.lib.workflow import Workflow
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
from env.workflow_scheduling_v3.lib.simsetting import Setting
from env.workflow_scheduling_v3.lib.processDAG import compute_initial_earliest_finish_times, update_node_weight_and_compute_finish_times
from env.workflow_scheduling_v3.lib.communication import CommunicationModel

vmidRange = 10000

def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)

def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):

    def __init__(self, args):

        self.set = Setting(args)
        self.baseHEFT = None
        self.trainSet = args["trainSet"]
        self.GENindex = None
        self.indEVALindex = None       
        self.TaskRule = None                # input th task selection rule here, if has
        # self.validSet = args["Validation"]

        if self.set.is_wf_trace_record:
            self.df = {}
            __location__ = os.path.join(os.getcwd(), "Saved_Results")
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s.csv' % (args["algo"],  args["seed"], args["arrival rate"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'VM ID', 'VM speed', 'Price', 'VM Rent Start Time', 'VM Rent End Time', 'VM Pending Index' ]) # 6 + 6 + 6 columns     

        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(6 + self.set.history_len,))
        self.action_space = gym.spaces.Discrete(n=100)  # n is a placeholder
        
        # ============ 通信模型初始化 ============
        self.comm_aware = getattr(configs, 'comm_aware', 0)
        self.comm_model = None
        if self.comm_aware:
            self.comm_model = CommunicationModel(configs)
            self.comm_model._randomize_bw = bool(getattr(configs, 'randomize_bandwidth', 0))
            self.comm_model.initialize_network(
                num_nodes=self.set.vmNum,
                topology=getattr(configs, 'network_topology', 'mesh')
            )
            # 设置干扰等级
            self.comm_model.set_interference_level(
                getattr(configs, 'interference_level', 0.0)
            )
        


    def close(self):
        print("Environment id %s is closed" % (self.set.envid))


    def _init(self):

        # 1) 初始化任务特征表Set record 12 features
        columns = ['Workflow ID', 'Task Index', 'Task Size', 'Wd2End', 'RemainWd in Wf', 'Task Schedule status', 'Task Ready Time',\
                   'Task Waiting Time', 'Task Execute Time', 'Task Finish Time', 'VM ID', 'VM speed', 'VM Process Index', 'VM Utilization']
        self.df = pd.DataFrame(columns=columns)        
        self.appSubDeadline = {}  # { app: {task: task_sub_deadline} } used as a state feature
        self.usr_queues = []            # [usr1:[workflows, ...], usr2:[workflows, ...]], e.g., user1 stores 30 workflows
        self.finishWf_queues = []
        self.activeWf_queues = []       # all complete and on-going workflows
        self.vm_queues = []             # [VM1, VM2, ...] each VM is a class
        self.vm_queues_id = []          # the vmIndex of each VM in self.vm_queues
        self.vm_queues_cpu = []
        self.vm_queues_utl = np.zeros((self.set.vmNum))
        self.usrNum = self.set.usrNum   ## useless for one usr
        self.dcNum = self.set.dcNum     ## useless for one cloud
        self.wfTypes = self.set.wfTypes
        self.wfNum= self.set.WorkflowNum
        self.vmNum = self.set.vmNum
        self.vmEachIntNum = self.set.vmEachIntNum       
        self.VMtypeNum = len(self.set.dataset.vmVCPU) ## number of VM types
        self.numTimestep = 0            # indicate how many timesteps have been processed
        self.completedWF = 0
        self.VMrentInfos = {}           # {VMid: [rent start time, rent end time]}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wfTypes, self.set.history_len)) 
        self.wfIndex = 0
        self.vmIndex = 10000            # labeled begin with 10000
        self.usrcurrentTime = np.zeros(self.usrNum)  # Used to record the current moment of the user
        self.dcCurrentTime = np.zeros(self.dcNum) 
        self.remainWrfNum = 0           # Record the number of packets remained in VMs 
        self.VMexecHours = 0        # The sum of processing time on all VMs
        self.VMidleTime = {}            ## Record the sum of the VM's current idle time to calculate VM utilization, += max(machine time, current time)      

        # IMPORTANT: used to get the ready task for the next time step
        self.firstvmWrfLeaveTime = []   # Record the current timestamp on each VM
        self.firstusrWrfGenTime = np.zeros(self.usrNum)  # Arrival time of the first inactive workflow in each user's workflow set
        self.vmLatestTime = {}

        self.uselessAllocation = 0
        # self.VMtobeRemove = None
        self.all_flowTime = np.zeros((self.wfNum))
        self.usr_respTime = np.zeros((self.usrNum, self.wfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wfTypes)) 
        self.usr_sent_pktNum = np.zeros((self.usrNum, self.dcNum))

        self.totalTaskSize = 0
        self.cumulative_reward_M = 0
        self.returns_M = 0
        self.maxVms = 0

        # ===== 拥塞辅助指标桶（episode 级，供 §5 主表 / §5.7 敏感性使用） =====
        # 与 reward_shaping 解耦，HEFT_comm 基线评测也会产出
        self._cong_stats = {
            'cong_extra_sum': 0.0,        # Σ_t max(0, t_cong - t_base)，秒
            'cong_extra_steps': 0,        # 触发拥塞的决策步数
            'hot_node_load_max': 0,       # max_t max_i ℓ_i(t)
            'hot_node_load_series': [],   # 每决策步的 max_i ℓ_i(t)，用于 p95
            'link_load_max': 0,           # max_t max_(s,d) link_transfers(s,d)
            'burst_step_count': 0,        # ready 任务 ≥ 6 的决策步数
            'total_decision_steps': 0,
            'task_wait_time_sum': 0.0,    # 累计任务等待时间（基于 workflow.get_waitingTime）
        }

        # 2) 上传所有工作流及其到达时间到 'self.firstusrWrfGenTime'
        # upload all workflows with their arrival time to the 'self.firstusrWrfGenTime'
        for i in range(self.usrNum):
            self.usr_queues.append(SimQueue())
            workflowsIDs = self.args.valid_dataset[self.GENindex][self.indEVALindex]
            arrTimes = self.args.arr_times[self.GENindex][self.indEVALindex]
            for wfID,arrT in zip(workflowsIDs,arrTimes):
                self.workflow_generator(i, wfID, arrT)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstWfEnqueueTime() 

            # 3) 生成各类型 VM 并添加到 'self.vm_queues'
            for vmID in range(len(self.set.dataset.vmVCPU)):
                self.vm_generator(0, vmID)

        # 4) 选出“下一步要处理的事件”：是新 workflow 的到达，还是某个 VM 完成任务
        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr() 
        self.PrenextTimeStep = self.nextTimeStep
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstWf() # obtain the root task of the first workflow in the self.nextUsr
        temp = self.nextWrf.get_allnextTask(self.finishTask)   # Get all real successor tasks of the virtual workflow root task
                
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]
        if len(temp) > 1:  # the next task has parallel successor tasks
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.check_add_newWf2df()
        self.stat = Stats(self.set)


    def check_add_newWf2df(self):
        if self.finishTask==None and self.nextWrf.appArivalIndex not in self.df['Workflow ID'].values:
            self.activeWf_queues.append(self.nextWrf)
            if configs.require_estimated_features in [1,2]:
                temp_earliest_finish_times = compute_initial_earliest_finish_times(self.nextWrf.app, self.nextWrf.generateTime)
            for node in self.nextWrf.app.nodes:
                if configs.require_estimated_features in [1,2]: # True
                    new_row = {'Workflow ID':self.nextWrf.appArivalIndex, 
                            'Task Index': node,
                                'Task Size':self.nextWrf.get_taskSize(node), 
                                'Wd2End':self.nextWrf.get_allsucc2end(node),
                                'RemainWd in Wf':self.nextWrf.remainWorkload, 
                                'Task Schedule status':3, 
                                'Task Ready Time': None, 'Task Execute Time': self.nextWrf.app.nodes[node]['estimatedET'], 
                                'Task Waiting Time':None, 'Task Finish Time': temp_earliest_finish_times[node],
                                'VM ID':None, 'VM speed': self.set.dataset.meanCPU, 'VM Process Index':None,\
                                'VM Utilization': None} #np.mean(self.vm_queues_utl)}
                else:
                    new_row = {'Workflow ID':self.nextWrf.appArivalIndex, 
                            'Task Index': node,
                                'Task Size':self.nextWrf.get_taskSize(node), 
                                'Wd2End':self.nextWrf.get_allsucc2end(node),
                                'RemainWd in Wf':self.nextWrf.remainWorkload, 
                                'Task Schedule status':3, 
                                'Task Ready Time': None, 'Task Execute Time': None, 
                                'Task Waiting Time':None, 'Task Finish Time': None,
                                'VM ID':None, 'VM speed': None, 'VM Process Index':None,\
                                'VM Utilization': None}                    
                # self.df = pd.concat([self.df, pd.DataFrame([new_row], columns=self.df.columns)], ignore_index=True)
                self.df.loc[len(self.df)] = new_row
                
            self.update_ready_tasks(self.nextWrf, self.finishTask)

    def update_ready_tasks(self, app, task):
        temp = app.get_allnextTask(task)
        for node in temp:   ## ready tasks
            parentTasks = app.get_allpreviousTask(node)
            if len(parentTasks) == len(app.completeTaskSet(parentTasks)):
                self.df.loc[(self.df['Workflow ID'] == app.appArivalIndex) & (self.df['Task Index'] == node), 
                        ['Task Schedule status', 'Task Ready Time', 'RemainWd in Wf']] = [2, self.nextTimeStep, app.remainWorkload] 

    # Generate one workflow at one time
    def workflow_generator(self, usr, wfID, nextarrT):

        wfDAG = self.set.dataset.wset[wfID]


        self.remainWrfNum += 1
        # add workflow deadline to the workflow
        Wf = Workflow(self.usrcurrentTime[usr], wfDAG, wfID, usr, self.set.dataset.wsetSlowestT[wfID],\
                      self.set.dataset.wsetTotProcessTime[wfID], self.wfIndex) # ,self.set.dueTimeCoef[usr, appID])

        self.usr_queues[usr].enqueue(Wf, self.usrcurrentTime[usr], None, usr, 0) # None means that workflow has not started yet
        self.usrcurrentTime[usr] = nextarrT
        self.wfNum-= 1
        self.wfIndex +=1


    def vm_generator(self, dcID, vmTypeID):

        for _ in range(self.set.vmEachIntNum):
            self.vmIndex+=1
            newVM = VM(self.vmIndex, self.set.dataset.vmVCPU[vmTypeID], dcID, self.set.dataset.datacenter[dcID][0], self.dcCurrentTime[dcID], self.TaskRule)
            self.vm_queues.append(newVM)
            self.firstvmWrfLeaveTime.append(newVM.get_firstTaskDequeueTime()) #new VM is math.inf
            self.vm_queues_id.append(self.vmIndex)
            self.vm_queues_cpu.append(self.set.dataset.vmVCPU[vmTypeID]) 
            self.VMrentInfos[self.vmIndex] = [self.vmIndex, self.set.dataset.vmVCPU[vmTypeID],  self.set.dataset.vmPrice[self.set.dataset.vmVCPU[vmTypeID]], 
                                        self.dcCurrentTime[dcID]]     


    def reset(self, seed1=0, seed2=0):
        random.seed(configs.env_seed)
        np.random.seed(configs.env_seed)
        self.GENindex = seed1
        self.indEVALindex = seed2   ## each individual is evaluated on only one problem instance
        self._init()
        self.PrenextWrf , self.PrenextTask = self.nextWrf, self.nextTask
        self.update_VMrentInfos()
        self.PreMakespan =  max(self.vmLatestTime.values())
        if self.comm_model and hasattr(self.comm_model, 'reset_transfers'):
            self.comm_model.reset_transfers()
            if getattr(self.comm_model, '_randomize_bw', False):
                _key = f"bw|{int(self.GENindex)}|{int(self.indEVALindex)}".encode("utf-8")
                bw_seed = int.from_bytes(hashlib.md5(_key).digest()[:4], "big") % (2**31)
                rng_state = np.random.get_state()
                np.random.seed(bw_seed)
                self.comm_model.initialize_network(
                    self.set.vmNum, getattr(configs, 'network_topology', 'mesh'))
                np.random.set_state(rng_state)
            else:
                self.comm_model.initialize_network(
                    self.set.vmNum, getattr(configs, 'network_topology', 'mesh'))
        self._hb_interval = int(getattr(configs, 'heartbeat_interval', 0))
        self._hb_step_count = 0
        self._cached_node_loads = {}  # node_id → cached load
        self._cached_congestion_level = None
        self._cached_comm_graph = None
        self._cached_bandwidth_matrix = None
        self._cached_hop_matrix = None
        if self._hb_interval > 0:
            self._update_cached_loads()


    def input_task_rule(self, rule):
        self.TaskRule = rule

    def generate_vmid(self):
        vmIndex = np.random.randint(vmidRange, size=1)[0]
        while vmIndex in self.vm_queues_id:
            vmIndex = np.random.randint(vmidRange, size=1)[0]
        return vmIndex

    def get_nextWrfFromUsr(self):       # Select the User with the smallest timestamp
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstWfTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstWfTime     # Returns the user and arrival time of the minimum arrival time of the workflow in the current User queue.

    def get_nextWrfFromVM(self):        # Select the machine with the smallest timestamp
        if len(self.firstvmWrfLeaveTime) > 0:
            vmInd = np.argmin(self.firstvmWrfLeaveTime)
            firstPktTime = self.firstvmWrfLeaveTime[vmInd]
            return vmInd, firstPktTime  # Returns vm-id and the minimum end time of the current VM
        else:
            return None, math.inf

    def get_nextTimeStep(self):

        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromVM() 
        if tempnextTimeStep > tempnextTimeStep1:  # task ready time > VM minimum time
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
                                        # The next step is to process the VM and update it to the timestep of the VM.
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.vm_queues[self.nextUsr].get_firstDequeueTask() # Only returns time, does not process task
        else:  # tempnextTimeStep <= tempnextTimeStep1
            if tempnextTimeStep == math.inf:   ## tempnextTimeStep：when self.usr_queues.queue is []
                self.nextTimeStep = None       ## tempnextTimeStep1：when self.firstvmWrfLeaveTime is []
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep # Next step is to process user & Update to user's timeStep
                self.nextisUsr = True    # Activate new Workflow from Usr_queue
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstWf() # The current first task in the selected user

        self.check_add_newWf2df()

    def update_VMrentInfos(self):

        for i, vm in zip(self.VMrentInfos, self.vm_queues):
            self.VMrentInfos[i][-1] = vm.currentTimeStep
            self.vmLatestTime[i] = vm.currentTimeStep + vm. pendingTaskTime
        for i, vm in enumerate(self.vm_queues):
            self.vm_queues_utl[i] = vm.get_vmUtilization(self.nextTimeStep)


    def step(self, action):


        mspan_entrance = float(self.PreMakespan)

        complete1Wf = False
        currentReward = 0

        selectedVMind = action

        # ===== 拥塞辅助指标：决策时刻采样（节点负载 / 链路负载 / burst） =====
        try:
            self._cong_stats['total_decision_steps'] += 1
            if self.comm_aware and self.comm_model is not None:
                _t_dec = self.PrenextTimeStep
                if _t_dec is not None and _t_dec != math.inf:
                    _load_vec = self.comm_model.get_node_load_vector(_t_dec)
                    _cur_max_node = int(_load_vec.max()) if _load_vec.size > 0 else 0
                    if _cur_max_node > self._cong_stats['hot_node_load_max']:
                        self._cong_stats['hot_node_load_max'] = _cur_max_node
                    self._cong_stats['hot_node_load_series'].append(_cur_max_node)

                    _link_mat = self.comm_model.get_link_congestion_matrix(_t_dec)
                    _cur_max_link = int(_link_mat.max()) if _link_mat.size > 0 else 0
                    if _cur_max_link > self._cong_stats['link_load_max']:
                        self._cong_stats['link_load_max'] = _cur_max_link
            # burst 度量：当前 ready 任务数 ≥ 6
            if isinstance(self.df, pd.DataFrame) and 'Task Schedule status' in self.df.columns:
                _ready_now = int((self.df['Task Schedule status'] == 2).sum())
                if _ready_now >= 6:
                    self._cong_stats['burst_step_count'] += 1
        except Exception:
            pass

        if self._hb_interval > 0:
            self._hb_step_count += 1
            if self._hb_step_count >= self._hb_interval:
                self._update_cached_loads()
                self._hb_step_count = 0

        # reward_M = 0
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep      
        self.numTimestep += 1  ## useless for GP

        # 1) 把当前候选任务派到 selectedVM
        # dispatch nextTask to selectedVM and update the wrfLeaveTime on selectedVM 
        parentTasks = self.PrenextWrf.get_allpreviousTask(self.PrenextTask)
        if len(parentTasks) == len(self.PrenextWrf.completeTaskSet(parentTasks)): # all its predecessor tasks have been done, just double-check
            comm_delay = 0.0
            edge_costs = None
            if self.comm_aware and self.comm_model is not None:
                if hasattr(self.comm_model, 'channel_contention') and self.comm_model.channel_contention:
                    # Compute every parent edge exactly once. The same sampled
                    # cost drives both task readiness and link registration.
                    edge_costs = self._compute_parent_comm_costs(
                        self.PrenextTask, selectedVMind, self.PrenextWrf,
                        self.PrenextTimeStep)
                    comm_delay = max((item[3] for item in edge_costs), default=0.0)
                else:
                    comm_delay = self.get_communication_time(self.PrenextTask, selectedVMind, self.PrenextWrf)
                if comm_delay > 0 and hasattr(self.comm_model, 'register_transfer'):
                    self._register_per_link_transfers(
                        self.PrenextTask, selectedVMind,
                        self.PrenextWrf, self.PrenextTimeStep,
                        edge_costs=edge_costs)
            effective_enqueue_time = self.PrenextTimeStep + comm_delay

            # 始终统计 congestion_extra（与 reward shaping 解耦），HEFT_comm 基线也产出
            if comm_delay > 0 and self.comm_aware and self.comm_model is not None:
                base_delay = self.get_communication_time(
                    self.PrenextTask, selectedVMind, self.PrenextWrf)
                congestion_extra = max(0.0, comm_delay - base_delay)
                if congestion_extra > 0:
                    self._cong_stats['cong_extra_sum'] += float(congestion_extra)
                    self._cong_stats['cong_extra_steps'] += 1

            processTime =  self.vm_queues[selectedVMind].task_enqueue(self.PrenextTask, effective_enqueue_time, self.PrenextWrf)
            self.VMexecHours += processTime/3600                                                                                                              
            self.firstvmWrfLeaveTime[selectedVMind] = self.vm_queues[selectedVMind].get_firstTaskDequeueTime() # return currunt timestap on this machine

            # 如果需要估计特征，更新 DAG 上所有任务的估计完成时间
            # Update the allocation infos of self.PrenextTask
            if configs.require_estimated_features in [1,2]:
                temp_earliest_finish_times = update_node_weight_and_compute_finish_times(self.PrenextWrf.app, self.PrenextWrf.generateTime,\
                                                                                        self.PrenextTask, processTime, self.PrenextWrf.actual_finish_time)
                for node, estimates in temp_earliest_finish_times.items():
                    self.df.loc[
                        (self.df['Workflow ID'] == self.PrenextWrf.appArivalIndex)
                        & (self.df['Task Index'] == node),
                        'Task Finish Time',
                    ] = estimates
            # 更新当前任务在 df 中的记录                     
            self.df.loc[(self.df['Workflow ID'] == self.PrenextWrf.appArivalIndex) & (self.df['Task Index'] == self.PrenextTask), 
                    ['RemainWd in Wf', 'Task Schedule status', 'Task Ready Time', 'Task Execute Time', 'Task Waiting Time',\
                    'VM ID', 'VM Process Index', 'VM speed', 'VM Utilization']]\
                    = [self.PrenextWrf.remainWorkload, 0, self.PrenextWrf.get_readyTime(self.PrenextTask), processTime,\
                        self.PrenextWrf.get_waitingTime(self.PrenextTask)] + self.PrenextWrf.pendingIndexOnDC[self.PrenextTask]    # + [VM ID', 'VM Process Index', 'VM speed']
            
            self.totalTaskSize += self.PrenextWrf.get_taskSize(self.PrenextTask)

        # 2) 根据 isDequeue 决定是从用户队列还是 VM 队列弹出“已分配的那个节点”
        # 2) Dequeue nextTask
        if self.isDequeue:      # True: the nextTask should be popped out 
            if self.nextisUsr:  # True: the nextTask to be deployed comes from the user queue
                self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                _, _ = self.usr_queues[self.PrenextUsr].dequeue() # Here is the actual pop-up of the root task 
                self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstWfEnqueueTime() 
                                                            # Updated with the arrival time of the next workflow
                self.usr_sent_pktNum[self.PrenextUsr][self.vm_queues[selectedVMind].get_relativeVMloc()] += 1
                self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_wfType(), self.nextWrf.get_generateTime()) 
            else:               # the nextTask to be deployed comes from the vm queues
                _, _ = self.vm_queues[self.PrenextUsr].task_dequeue() # Here nextTask actually starts to run
                self.firstvmWrfLeaveTime[self.PrenextUsr] = self.vm_queues[self.PrenextUsr].get_firstTaskDequeueTime()
                                                            # Update the current TimeStamp in this machine

        # 3) 处理任务完成 & 工作流完成，寻找下一候选任务
        # 3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)   # all successor tasks of the current self.finishTask
                                    # and one successor task has already enqueued

        if len(temp_Children_finishTask) > 0:
            self.dispatchParallelTaskNum += 1

        while True: 
            # 3.1 如果当前 finishTask 是 workflow 的最后一个任务
            # self.nextWrf is completed
            while len(temp_Children_finishTask) == 0:  # self.finishTask is the final task of self.nextWrf
                
                if self.nextisUsr:  # for double-check: Default is False
                    # Because it corresponds to self.finishTask, if temp==0, it means it cannot be entry tasks
                    print('self.nextisUsr maybe wrong')
                _, app = self.vm_queues[self.nextUsr].task_dequeue()  
                self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime() 
                        # If there is no task on the VM, math.inf will be returned
                if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):     # self.nextWrf has been completed
                    respTime = self.nextTimeStep - self.nextWrf.get_generateTime()
                    self.usr_respTime[app.get_originDC()][app.get_wfID()] = respTime/app.totalProcessTime
                    self.usr_received_wrfNum[app.get_originDC()][app.get_wfType()] += 1                    
                    self.completedWF += 1
                    self.remainWrfNum -= 1
                    # ddl_penalty = self.calculate_penalty(app, respTime)
                    # self.SLApenalty += ddl_penalty
                    # self.record_a_completed_workflow(ddl_penalty)
                    # self.all_flowTime.append(respTime)
                    self.all_flowTime[app.appArivalIndex] = respTime

                    complete1Wf = True
                    currentReward -= respTime 
                    # currentReward -= max(self.set.dataset.vmVCPU) * respTime/self.set.dataset.wsetSlowestT[app.appID] 
                    
                    self.finishWf_queues.append([app, self.df[self.df['Workflow ID'] == app.appArivalIndex]])
                    self.df = self.df[self.df['Workflow ID'] != app.appArivalIndex]
                    self.df = self.df.reset_index(drop=True)

                    del app, self.nextWrf                   

                self.get_nextTimeStep()
                if self.nextTimeStep is None:
                    break
                # self.update_VMRemain_infos()
                # self.remove_expired_VMs()                
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                            ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']]\
                            = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload] 
                self.update_ready_tasks(self.nextWrf, self.finishTask)                  
                temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)              

            if self.nextTimeStep is None:
                break

            # 3.2 如果当前 finishTask 不是 workflow 的最后一个任务，则需要分配并行任务
            # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
            if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                to_be_next = None
                while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                    temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                    temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                    if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) >0:
                        self.dispatchParallelTaskNum += 1
                    else: 
                        to_be_next = temp_nextTask
                        break

                if to_be_next is not None: 
                    self.nextTask = to_be_next
                    if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                        self.isDequeue = False
                    else:
                        self.isDequeue = True
                    break

                else: # Mainly to loop this part
                    _, _ = self.vm_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                    self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                    self.get_nextTimeStep() 
                    # self.update_VMRemain_infos()
                    # self.remove_expired_VMs()                        
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                    self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                                ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']]\
                              = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload]      
                    self.update_ready_tasks(self.nextWrf, self.finishTask)                
                    self.dispatchParallelTaskNum = 0                     
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

            else: # i.e., len(temp_Children_finishTask)<=self.dispatchParallelTaskNum
                # self.nextTask is the last imcompleted successor task of the self.finishTask
                if not self.isDequeue:      # Defaults to True
                    print('self.isDequeue maybe wrong')      
                self.get_nextTimeStep()
                # self.update_VMRemain_infos()
                # self.remove_expired_VMs()                    
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                        ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']] \
                        = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload]      
                self.update_ready_tasks(self.nextWrf, self.finishTask)
                self.dispatchParallelTaskNum = 0 # Restart recording the number of successor tasks of self.finishTask
                if self.nextTimeStep is not None:
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        # 4) 更新到达率历史、VM 租期信息、标记“下一候选任务”为状态
        self.notNormalized_arr_hist = self.stat.update_arrival_rate_history() ## useless for GP
        if self.nextTimeStep is not None:
            self.update_VMrentInfos()
            self.df.loc[
                (self.df['Workflow ID'] == self.nextWrf.appArivalIndex)
                & (self.df['Task Index'] == self.nextTask),
                'Task Schedule status',
            ] = 1  # to_be_allocated
        self.Makespan =  max(self.vmLatestTime.values())
        
        # 5) 判断是否 episode 结束
        done = False
        if self.remainWrfNum == 0:
            if len(self.firstvmWrfLeaveTime) == 0:
                done = True
            elif self.firstvmWrfLeaveTime[0] == math.inf and self.firstvmWrfLeaveTime.count(self.firstvmWrfLeaveTime[0]) == len(self.firstvmWrfLeaveTime):
                done = True

        # 6) 记录 makespan 统计
        reward_M = self.PreMakespan - self.Makespan
        self.cumulative_reward_M += reward_M
        self.PreMakespan = self.Makespan
        self.PrenextWrf , self.PrenextTask = self.nextWrf, self.nextTask

        if done:
            # reward_M = -self.VMcost-self.SLApenalty
            self.returns_M = self.Makespan
            # 汇总拥塞辅助指标（标量化 series 为 max/p95），供 evaluator/CSV 使用
            try:
                _series = self._cong_stats.get('hot_node_load_series', [])
                if len(_series) > 0:
                    _arr = np.asarray(_series, dtype=np.float32)
                    _hot_p95 = float(np.percentile(_arr, 95))
                    _hot_mean = float(_arr.mean())
                else:
                    _hot_p95 = 0.0
                    _hot_mean = 0.0
                _total_steps = max(1, int(self._cong_stats.get('total_decision_steps', 1)))
                _burst_ratio = float(self._cong_stats.get('burst_step_count', 0)) / _total_steps
                _cong_step_ratio = float(self._cong_stats.get('cong_extra_steps', 0)) / _total_steps
                _cong_summary = {
                    'cong_extra_sum': float(self._cong_stats.get('cong_extra_sum', 0.0)),
                    'cong_extra_steps': int(self._cong_stats.get('cong_extra_steps', 0)),
                    'cong_extra_step_ratio': _cong_step_ratio,
                    'hot_node_load_max': int(self._cong_stats.get('hot_node_load_max', 0)),
                    'hot_node_load_p95': _hot_p95,
                    'hot_node_load_mean': _hot_mean,
                    'link_load_max': int(self._cong_stats.get('link_load_max', 0)),
                    'burst_step_ratio': _burst_ratio,
                    'total_decision_steps': _total_steps,
                }
            except Exception:
                _cong_summary = {}
            self.episode_info = {'makespanOfWf':self.usr_respTime, 'processRate':self.Makespan/self.totalTaskSize, 'makespan':self.returns_M,\
                                  'return':self.cumulative_reward_M, 'Infos':self.finishWf_queues, 'maxVMs': self.maxVms,\
                                    'responseTime': np.mean(self.all_flowTime),\
                                    'cong_stats': _cong_summary}
            # print(self.maxVms)

        # 7) 返回 step 的即时奖励
        # 通信延迟已在 task_enqueue 阶段加入 enqueueTime，flowtime 自然包含通信代价
        reward = currentReward
        self.maxVms = len(self.df) #max(self.maxVms, len(self.df))

        return reward, done


    # calculate the total VM cost during an episode
    def update_VMcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.VMcost += temp * self.set.dataset.vmPrice[cpu]      # (self.set.dataset.datacenter[dc][-1])/2 * cpu
        self.VMrentHours += temp


    def calculate_penalty(self, app, respTime):
        appID = app.get_wfType()
        threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold,5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            return 1+self.set.dataset.wsetBeta[appID]*(respTime-threshold)/3600


    def add_self_loops(self, dag):

        selfEdges = []
        for node in dag.nodes():
            selfEdges.append([node, node])

        return np.array(selfEdges).T

    def state_info_construct1(self):  #
        ## All VM Utilization values ​​with empty vm id are replaced with the real-time mean_utl value.
        # 1) 填充 VM Utilization
        if configs.require_estimated_features == 1:
            self.df.loc[self.df['VM ID'].isnull(), 'VM Utilization'] = np.mean(self.vm_queues_utl) 
        # Workflow infos: wf_features, wf_edges, wf_masks, candidate_taskID 
        # 2) 任务特征 wf_features，包括：任务调度状态、任务大小、截止时间、执行时间、完成时间、VM 速度、VM 利用率
        selected_features = ['Task Schedule status','Task Size','Wd2End', 'Task Execute Time',\
                             'Task Finish Time',  'VM speed', 'VM Utilization'] #
        wf_features = self.df.loc[:, selected_features].values
        
        # ---- 相对化 Task Finish Time (index=4): 减去各workflow的到达时间 ----
        wf_ids = self.df['Workflow ID'].values
        gen_times = np.array([self.activeWf_queues[wid].generateTime
                              if wid in self.activeWf_queues else 0
                              for wid in wf_ids], dtype=np.float64)
        ft_col = 4
        wf_features[wf_features == None] = 0
        ft_vals = wf_features[:, ft_col].astype(np.float64)
        non_zero = ft_vals != 0
        ft_vals[non_zero] -= gen_times[non_zero]
        wf_features[:, ft_col] = ft_vals
        
        if self.comm_aware and self.comm_model is not None:
            n_tasks = wf_features.shape[0]
            padding = np.zeros((n_tasks, 4), dtype=np.float32)
            # Heartbeat staleness symmetry: when _hb_interval>0, agent must
            # observe the same stale view of node load that HEFT_comm uses
            # via _cached_node_loads (see HEFT_comm_aware/_update_cached_loads).
            # Otherwise the RL actor is unfairly given oracle real-time info.
            _use_cached_obs = (self._hb_interval > 0
                               and len(self._cached_node_loads) > 0)
            if _use_cached_obs:
                node_loads = np.array(
                    [self._cached_node_loads.get(i, 0.0)
                     for i in range(self.set.vmNum)],
                    dtype=np.float32,
                )
                cong = float(self._cached_congestion_level)
            else:
                cong = self.comm_model.get_congestion_level(self.nextTimeStep)
                node_loads = (self.comm_model.get_node_load_vector(self.nextTimeStep)
                              if hasattr(self.comm_model, 'get_node_load_vector')
                              else None)
            padding[:, 0] = cong
            if node_loads is not None:
                n_vm = max(1, len(self.vm_queues))
                parent_tasks = self.nextWrf.get_allpreviousTask(self.nextTask)
                max_parent_load = 0.0
                n_remote = 0
                for pt in (parent_tasks or []):
                    if pt in self.nextWrf.processDC:
                        pvm = self._get_vm_index_by_id(self.nextWrf.processDC[pt])
                        if pvm < len(node_loads):
                            max_parent_load = max(max_parent_load, float(node_loads[pvm]))
                        n_remote += 1
                totalTasks_tmp = 0
                cand_idx = None
                for wfID in self.df['Workflow ID'].unique():
                    if wfID == self.nextWrf.appArivalIndex:
                        cand_idx = totalTasks_tmp + self.nextTask
                    totalTasks_tmp += self.activeWf_queues[wfID].taskNum
                if cand_idx is not None and cand_idx < n_tasks:
                    padding[cand_idx, 1] = min(max_parent_load / max(1, n_vm), 1.0)
                    padding[cand_idx, 2] = n_remote / max(1, n_vm)
                    padding[cand_idx, 3] = max_parent_load
            wf_features = np.hstack([wf_features, padding])
        
        # 3) 构造工作流 DAG 边 wf_edges / 掩码 wf_masks / candidate_taskID
        wf_edges = []
        wf_masks = []                  
        totalTasks = 0
        for wfID in self.df['Workflow ID'].unique():
            if configs.remove_completed==0:
                array1 = np.array(self.activeWf_queues[wfID].tempDAG.edges()).T
                if array1.size == 0:
                    temp_edges = self.add_self_loops(self.activeWf_queues[wfID].tempDAG) + totalTasks
                else:
                    temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].tempDAG)), axis=1) + totalTasks
            else:
                array1 = np.array(self.activeWf_queues[wfID].app.edges()).T
                temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].app)), axis=1) + totalTasks            
            wf_edges.append(temp_edges)
            if configs.remove_completed==0:
                wf_masks.append(self.activeWf_queues[wfID].mask)
            else:
                wf_masks.append(np.full(shape=self.activeWf_queues[wfID].app.number_of_nodes(), fill_value=1, dtype=bool))
            if wfID == self.nextWrf.appArivalIndex:
                candidate_taskID = totalTasks + self.nextTask
            totalTasks += self.activeWf_queues[wfID].taskNum
        wf_edges = np.concatenate(wf_edges, axis=1)
        wf_masks = np.concatenate(wf_masks, dtype=bool) 
        wf_edges = np.flip(wf_edges, axis=0)    

        # 4) VM 特征 vm_features & VM 队列的边 vm_edges
        vm_features = []
        vm_edges = []
        # vm_masks = []
        # totalNodes = 0 
        vm_lastIdx = []  
        for i,vm in enumerate(self.vm_queues):
            temp_vm_edges = vm.get_VM_virtual(self.nextTimeStep, False) 
            if len(temp_vm_edges) > 0:
                temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[0][0]) & (self.df['Task Index'] == temp_vm_edges[0][1])].tolist()
                vm_lastIdx.append(temp_idx[0])
                for k in range(1,len(temp_vm_edges)):  
                    temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[k][0]) & (self.df['Task Index'] == temp_vm_edges[k][1])].tolist()
                    wf_edges = np.column_stack([wf_edges, [vm_lastIdx[-1], temp_idx[0]]])
                    vm_lastIdx[-1] = temp_idx[0]
            else:
                vm_lastIdx.append([])
            # 准备要添加进wf_features的相应infos
            raw_finish = vm.get_taskFinishTime(self.nextTask, self.nextWrf)
            rel_finish = raw_finish - self.nextTimeStep if raw_finish else 0
            base_vm_features = [vm.get_taskExecuteTime(self.nextTask, self.nextWrf),
                               rel_finish, 
                               vm.cpu,
                               vm.get_vmUtilization(self.nextTimeStep)]
            
            if self.comm_aware and self.comm_model is not None:
                comm_time = self._observed_comm_time(
                    self.nextTask, i, self.nextWrf, self.nextTimeStep)
                avg_bandwidth = self._get_avg_bandwidth(i)
                avg_hops = self._get_avg_hops(i)
                congestion_level = self._observed_congestion_level(self.nextTimeStep)
                base_vm_features.extend([comm_time, avg_bandwidth, avg_hops, congestion_level])
            
            vm_features.append(base_vm_features)
            vm_edges.append([vm_lastIdx[-1], candidate_taskID] if isinstance(vm_lastIdx[-1],int) else [])

        wf_batchs = np.full((len(wf_features),), 0, dtype=np.int64)  

        comm_node_features, comm_edge_index, comm_edge_attr, vm_id_map = self._build_comm_graph_state()
        return (np.array([candidate_taskID]), wf_features.astype(np.float32), wf_edges, wf_masks, wf_batchs,\
                 np.array(vm_features), np.array(vm_edges, dtype=object), np.array(vm_lastIdx,dtype=object), np.array([]),
                 comm_node_features, comm_edge_index, comm_edge_attr, vm_id_map)




    

    def gp_feature_construct(self):

        '''
        features:
        0.	TS: task_size of a task
        1.	RW: remain workload in the workflow
        2.	ET: execute_time_period of a task on this VM
        3.	FT: finish_time = start_time + ET
        4.  CU: compute unit of a VM
        5.	UL: utilization of a VM 

        '''
        obs = np.zeros((self.vmNum, 6))
        features = self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) &\
                                (self.df['Task Index'] == self.nextTask), ['Task Size','Wd2End']] 
        obs[:,:2] = features.values

        for i,vm in enumerate(self.vm_queues):
            obs[i,2] = vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
            obs[i,3] = vm.vmLatestTime() + obs[i,2]   #get_taskFinishTime(self.nextTask, self.nextWrf)
            obs[i,4] = vm.cpu
            obs[i,5] = vm.get_vmUtilization(self.nextTimeStep)
 
        return obs



    def CPOP(self):
        """
        Critical Path on a Processor: 关键路径任务分配到最快VM，
        非关键路径任务用HEFT策略。
        """
        task_df = self.df[
            (self.df['Workflow ID'] == self.nextWrf.appArivalIndex)
            & (self.df['Task Index'] == self.nextTask)
        ]
        is_critical = False
        if len(task_df) > 0 and 'RemainWd in Wf' in task_df.columns:
            remain = task_df.iloc[0].get('RemainWd in Wf', 0)
            total = self.nextWrf.totalWorkload if hasattr(self.nextWrf, 'totalWorkload') else 1
            if total > 0 and remain / max(total, 1) > 0.5:
                is_critical = True

        if is_critical:
            exec_times = [vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
                          for vm in self.vm_queues]
            return int(np.argmin(exec_times))
        else:
            return self.HEFT()

    def HEFT(self):
        """
        HEFT启发式调度
        
        如果启用了通信感知(comm_aware=1)，则使用通信感知版本
        """
        if self.comm_aware and self.comm_model is not None:
            return self.HEFT_comm_aware()
        
        # 原始HEFT逻辑
        priorities = []
        for vm in self.vm_queues:
            priorities.append(vm.vmLatestTime() + vm.get_taskExecuteTime(self.nextTask, self.nextWrf) ) 
        min_indices = np.where(priorities == np.min(priorities))[0]

        return np.random.choice(min_indices)
    
    def HEFT_comm_aware(self):
        """
        通信感知的HEFT启发式
        当启用心跳延迟(heartbeat_interval>0)时，使用缓存的过时拥塞数据（模拟中心调度器的信息延迟）
        否则使用实时拥塞数据
        """
        priorities = []
        use_congestion = (hasattr(self.comm_model, 'channel_contention')
                          and self.comm_model.channel_contention)
        use_cached = self._hb_interval > 0
        
        for vm_idx, vm in enumerate(self.vm_queues):
            exec_time = vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
            
            if use_congestion:
                if use_cached:
                    comm_time = self.get_communication_time_congested_cached(
                        self.nextTask, vm_idx, self.nextWrf, self.nextTimeStep)
                else:
                    comm_time = self.get_communication_time_congested(
                        self.nextTask, vm_idx, self.nextWrf, self.nextTimeStep)
            else:
                comm_time = self.get_communication_time(
                    self.nextTask, vm_idx, self.nextWrf)
            
            vm_available = vm.vmLatestTime()
            ready_with_comm = self.nextTimeStep + comm_time
            eft = max(vm_available, ready_with_comm) + exec_time
            
            priorities.append(eft)
            
        min_indices = np.where(priorities == np.min(priorities))[0]
        return np.random.choice(min_indices)
    
    def get_communication_time(self, task, dst_vm_idx, workflow):
        """
        计算任务的通信时间
        
        Args:
            task: 任务ID
            dst_vm_idx: 目标VM索引
            workflow: 工作流对象
            
        Returns:
            最大通信时间（秒）
        """
        if not self.comm_aware or self.comm_model is None:
            return 0.0
            
        parent_tasks = workflow.get_allpreviousTask(task)
        if not parent_tasks:
            return 0.0
            
        max_comm_time = 0.0
        
        for parent in parent_tasks:
            # 获取父任务所在的VM
            if parent in workflow.processDC:
                parent_vm_id = workflow.processDC[parent]
                # 查找parent_vm在vm_queues中的索引
                parent_vm_idx = self._get_vm_index_by_id(parent_vm_id)
                
                # 如果父任务和当前任务在同一VM，无通信开销
                if parent_vm_idx == dst_vm_idx:
                    continue
                    
                # 获取边上的数据大小
                data_size = workflow.get_data_size(parent, task)
                
                if data_size > 0:
                    comm_cost = self.comm_model.get_communication_cost(
                        parent_vm_idx, dst_vm_idx, data_size
                    )
                    max_comm_time = max(max_comm_time, comm_cost)
                    
        return max_comm_time

    def _observed_comm_time(self, task, dst_vm_idx, workflow, current_time):
        """通信观测统一入口：H>0 使用最近 heartbeat 快照，H=0 使用实时状态。"""
        if not self.comm_aware or self.comm_model is None:
            return 0.0
        if getattr(self.comm_model, 'channel_contention', False):
            if self._hb_interval > 0:
                return self.get_communication_time_congested_cached(
                    task, dst_vm_idx, workflow, current_time)
            return self.get_communication_time_congested(
                task, dst_vm_idx, workflow, current_time)
        return self.get_communication_time(task, dst_vm_idx, workflow)

    def _observed_congestion_level(self, current_time):
        """返回策略当前可见的拥塞标量，与通信图/候选特征使用同一快照。"""
        if self._hb_interval > 0 and self._cached_congestion_level is not None:
            return float(self._cached_congestion_level)
        if self.comm_model is None:
            return 0.0
        return float(self.comm_model.get_congestion_level(current_time))

    def _compute_parent_comm_costs(self, task, dst_vm_idx, workflow, current_time):
        """一次 dispatch 计算并缓存所有跨 VM 父边通信代价。"""
        edge_costs = []
        for parent in workflow.get_allpreviousTask(task):
            if parent not in workflow.processDC:
                continue
            src_vm_idx = self._get_vm_index_by_id(workflow.processDC[parent])
            if src_vm_idx == dst_vm_idx:
                continue
            data_size = workflow.get_data_size(parent, task)
            if data_size <= 0:
                continue
            cost = self.comm_model.get_communication_cost_congested(
                src_vm_idx, dst_vm_idx, data_size, current_time)
            edge_costs.append((src_vm_idx, dst_vm_idx, data_size, float(cost)))
        return edge_costs

    def get_communication_time_congested(self, task, dst_vm_idx, workflow, current_time):
        """带信道拥塞的通信时间计算（实时拥塞，用于实际调度执行）"""
        if not self.comm_aware or self.comm_model is None:
            return 0.0
        parent_tasks = workflow.get_allpreviousTask(task)
        if not parent_tasks:
            return 0.0

        max_comm_time = 0.0
        for parent in parent_tasks:
            if parent in workflow.processDC:
                parent_vm_id = workflow.processDC[parent]
                parent_vm_idx = self._get_vm_index_by_id(parent_vm_id)
                if parent_vm_idx == dst_vm_idx:
                    continue
                data_size = workflow.get_data_size(parent, task)
                if data_size > 0:
                    cost = self.comm_model.get_communication_cost_congested(
                        parent_vm_idx, dst_vm_idx, data_size, current_time)
                    max_comm_time = max(max_comm_time, cost)
        return max_comm_time

    def get_communication_time_congested_cached(self, task, dst_vm_idx, workflow, current_time):
        """带心跳延迟的拥塞通信时间（使用缓存的过时拥塞数据，供HEFT/先验使用）"""
        if not self.comm_aware or self.comm_model is None:
            return 0.0
        if self._hb_interval <= 0:
            return self.get_communication_time_congested(task, dst_vm_idx, workflow, current_time)
        parent_tasks = workflow.get_allpreviousTask(task)
        if not parent_tasks:
            return 0.0

        max_comm_time = 0.0
        for parent in parent_tasks:
            if parent in workflow.processDC:
                parent_vm_id = workflow.processDC[parent]
                parent_vm_idx = self._get_vm_index_by_id(parent_vm_id)
                if parent_vm_idx == dst_vm_idx:
                    continue
                data_size = workflow.get_data_size(parent, task)
                if data_size > 0:
                    cost = self._comm_cost_with_cached_load(
                        parent_vm_idx, dst_vm_idx, data_size, current_time)
                    max_comm_time = max(max_comm_time, cost)
        return max_comm_time

    def _comm_cost_with_cached_load(self, src, dst, data_size_mb, current_time):
        """用缓存的节点负载计算通信代价（模拟心跳延迟）"""
        cm = self.comm_model
        if data_size_mb <= 0 or src == dst:
            return 0.0
        src_load = self._cached_node_loads.get(src, 0)
        dst_load = self._cached_node_loads.get(dst, 0)
        congestion_divisor = cm.congestion_multiplier(max(src_load, dst_load))

        obs_bw = self._cached_bandwidth_matrix if self._hb_interval > 0 and self._cached_bandwidth_matrix is not None else cm.bandwidth_matrix
        obs_hops = self._cached_hop_matrix if self._hb_interval > 0 and self._cached_hop_matrix is not None else cm.hop_matrix
        base_bw = obs_bw[src, dst]
        if base_bw == 0 or not np.isfinite(base_bw):
            return cm.get_communication_cost(src, dst, data_size_mb)
        noise = np.random.normal(0, cm.bandwidth_variance * cm.interference_level)
        interference_reduction = cm.interference_level * cm.ew_factor * base_bw
        effective_bw = base_bw * (1 - abs(noise)) - interference_reduction
        effective_bw = max(effective_bw, cm.get_path_bandwidth_floor(src, dst, obs_hops))
        effective_bw = effective_bw / max(congestion_divisor, 1e-6)
        transfer_time = (data_size_mb * 8) / max(effective_bw, 0.01)
        hops = int(obs_hops[src, dst])
        latency = (cm.base_latency + hops * cm.latency_per_hop) / 1000
        total_time = transfer_time + latency + cm.interference_level * 0.05
        effective_loss_rate = cm.packet_loss_rate * (1 + cm.interference_level * 5)
        success_prob = (1 - effective_loss_rate) ** hops
        expected_attempts = min(1 / max(success_prob, 1e-6), cm.max_retransmit)
        cost = total_time * expected_attempts
        if success_prob < 0.9:
            cost += (1 - success_prob) * 1.0
        return cost

    def _update_cached_loads(self):
        """更新心跳快照：节点负载、拥塞标量和通信图必须来自同一时刻。"""
        if self.comm_model is None:
            return
        t = getattr(self, 'nextTimeStep', None)
        if t is None or t == float('inf'):
            t = getattr(self, 'PrenextTimeStep', 0.0)
        for node_id in range(self.set.vmNum):
            self._cached_node_loads[node_id] = self.comm_model.get_node_radio_load(node_id, t)
        self._cached_congestion_level = float(self.comm_model.get_congestion_level(t))
        graph = self.comm_model.export_graph_tensors(t)
        self._cached_comm_graph = tuple(np.array(item, copy=True) for item in graph)
        self._cached_bandwidth_matrix = self.comm_model.bandwidth_matrix.copy()
        self._cached_hop_matrix = self.comm_model.hop_matrix.copy()

    def _register_per_link_transfers(self, task, dst_vm_idx, workflow,
                                     current_time, edge_costs=None):
        """为每条父→子链路注册传输；若已提供 edge_costs，则不重新采样。"""
        if edge_costs is None:
            edge_costs = self._compute_parent_comm_costs(
                task, dst_vm_idx, workflow, current_time)
        for src_vm_idx, dst_idx, _data_size, cost in edge_costs:
            if cost > 0:
                self.comm_model.register_transfer(
                    current_time + cost, src=src_vm_idx, dst=dst_idx)

    # ================================================================
    #  基线调度算法
    # ================================================================

    def _get_vm_index_by_id(self, vm_id):
        """根据VM ID获取在vm_queues中的索引"""
        try:
            return self.vm_queues_id.index(vm_id)
        except (ValueError, AttributeError):
            return 0

    def _get_avg_bandwidth(self, vm_idx):
        """
        获取VM到其他所有VM的平均带宽
        
        Args:
            vm_idx: VM在vm_queues中的索引
            
        Returns:
            平均带宽 (Mbps)，归一化到0-1范围
        """
        if self.comm_model is None or self.comm_model.bandwidth_matrix is None:
            return 1.0
        
        bw_matrix = self._cached_bandwidth_matrix if self._hb_interval > 0 and self._cached_bandwidth_matrix is not None else self.comm_model.bandwidth_matrix
        bandwidths = []
        for i in range(bw_matrix.shape[0]):
            if i != vm_idx:
                bw = bw_matrix[vm_idx, i]
                if bw > 0 and bw < float('inf'):
                    bandwidths.append(bw)
        
        if len(bandwidths) == 0:
            return 1.0
        
        # 归一化：除以基础带宽
        avg_bw = np.mean(bandwidths)
        return avg_bw / self.comm_model.base_bandwidth
    
    def _get_avg_hops(self, vm_idx):
        """
        获取VM到其他所有VM的平均跳数
        
        Args:
            vm_idx: VM在vm_queues中的索引
            
        Returns:
            平均跳数，归一化到0-1范围
        """
        if self.comm_model is None or self.comm_model.hop_matrix is None:
            return 0.0
        
        hop_matrix = self._cached_hop_matrix if self._hb_interval > 0 and self._cached_hop_matrix is not None else self.comm_model.hop_matrix
        hops = []
        for i in range(hop_matrix.shape[0]):
            if i != vm_idx:
                h = hop_matrix[vm_idx, i]
                if h < 100:  # 排除不可达
                    hops.append(h)
        
        if len(hops) == 0:
            return 0.0
        
        # 归一化：假设最大跳数为10
        avg_hops = np.mean(hops)
        return min(avg_hops / 10.0, 1.0)

    def _build_comm_graph_state(self):
        """构建通信图附加状态（append-only），传递当前仿真时间以获取动态拥塞特征。"""
        if not self.comm_aware or self.comm_model is None:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )
        if self._hb_interval > 0 and self._cached_comm_graph is not None:
            node_f, edge_idx, edge_attr = self._cached_comm_graph
            vm_id_map = np.arange(self.set.vmNum, dtype=np.int64)
            return (np.array(node_f, copy=True), np.array(edge_idx, copy=True),
                    np.array(edge_attr, copy=True), vm_id_map)
        current_time = getattr(self, 'nextTimeStep', None)
        if current_time is None or current_time == float('inf'):
            current_time = getattr(self, 'PrenextTimeStep', 0.0)
        node_f, edge_idx, edge_attr = self.comm_model.export_graph_tensors(current_time)
        vm_id_map = np.arange(self.set.vmNum, dtype=np.int64)
        return node_f, edge_idx, edge_attr, vm_id_map



    def _peft_future_cost(self, task, vm_idx, workflow, memo):
        """Return a PEFT-style optimistic cost-to-go after the current task.

        The recursion uses the current instance's fixed pairwise link rates and ignores
        future contention, as in an optimistic cost table. It is deliberately
        separate from the online EFT, which uses the observed congestion
        state. Results are memoized for the current workflow decision.
        """
        key = (int(task), int(vm_idx))
        if key in memo:
            return memo[key]
        successors = list(workflow.app.successors(task))
        if not successors:
            memo[key] = 0.0
            return 0.0
        cm = self.comm_model
        tail = 0.0
        for child in successors:
            data_size = float(workflow.get_data_size(task, child))
            best = float("inf")
            for child_vm_idx, child_vm in enumerate(self.vm_queues):
                if child_vm_idx == vm_idx or data_size <= 0 or cm is None:
                    comm = 0.0
                else:
                    bw = float(cm.bandwidth_matrix[vm_idx, child_vm_idx])
                    if not np.isfinite(bw) or bw <= 0:
                        comm = 0.0
                    else:
                        hops = int(cm.hop_matrix[vm_idx, child_vm_idx])
                        comm = (data_size * 8.0) / max(bw, 1e-9)
                        comm += (cm.base_latency + hops * cm.latency_per_hop) / 1000.0
                comp = child_vm.get_taskExecuteTime(child, workflow)
                value = comm + comp + self._peft_future_cost(
                    child, child_vm_idx, workflow, memo)
                best = min(best, value)
            tail = max(tail, best if np.isfinite(best) else 0.0)
        memo[key] = float(tail)
        return float(tail)

    def PEFT_comm(self):
        """Communication-aware PEFT-style optimistic-cost-table baseline.

        The current candidate's EFT uses the same communication cost and
        heartbeat observation as HEFT_comm. The optimistic tail is computed
        recursively over successors using fixed pairwise link rates and the fastest
        feasible successor placement, so it remains optimistic and does not
        reuse the learned policy.
        """
        if not self.vm_queues:
            return 0
        memo = {}
        priorities = []
        use_congestion = bool(
            self.comm_model is not None
            and getattr(self.comm_model, 'channel_contention', 0)
        )
        use_cached = self._hb_interval > 0
        for vm_idx, vm in enumerate(self.vm_queues):
            exec_time = vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
            if use_congestion:
                if use_cached:
                    comm_time = self.get_communication_time_congested_cached(
                        self.nextTask, vm_idx, self.nextWrf, self.nextTimeStep)
                else:
                    comm_time = self.get_communication_time_congested(
                        self.nextTask, vm_idx, self.nextWrf, self.nextTimeStep)
            elif self.comm_aware and self.comm_model is not None:
                comm_time = self.get_communication_time(
                    self.nextTask, vm_idx, self.nextWrf)
            else:
                comm_time = 0.0
            ready = self.nextTimeStep + comm_time
            eft = max(vm.vmLatestTime(), ready) + exec_time
            # Candidate-specific optimistic cost-table term.
            optimistic_tail = self._peft_future_cost(
                self.nextTask, vm_idx, self.nextWrf, memo)
            priorities.append(eft + optimistic_tail)
        min_indices = np.where(np.isclose(priorities, np.min(priorities)))[0]
        return int(np.random.choice(min_indices))
    


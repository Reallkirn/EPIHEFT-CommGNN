"""Evaluation-only interface to the original scheduling simulator."""
from env.workflow_scheduling_v3.lib.cloud_env_maxPktNum import cloud_simulator

class WFEnv(cloud_simulator):
    def __init__(self, args):
        config = dict(traf_type=args.traf_type,seed=args.env_seed,arr_rate=args.arr_rate,envid=0,
                      vm_types=args.vm_types,wf_size=args.wf_size,wf_num=args.wf_num,
                      trainSet=args.valid_dataset,each_vm_type_num=args.each_vm_type_num)
        super().__init__(config)
        self.name = args.env_name
        self.args = args
        self.train_or_test = False

    def reset(self):
        super().reset(self.args.GENindex,self.args.indEVALindex)
        self.step_curr = self.numTimestep
        return self.state_info_construct1()

    def step(self,action):
        reward,done=super().step(action)
        return (self.episode_info if done else self.state_info_construct1()),reward,done

    def resetGP(self):
        super().reset(self.args.GENindex,self.args.indEVALindex)
        return self.gp_feature_construct()

    def stepGP(self,action):
        reward,done=super().step(action)
        return (self.episode_info if done else self.gp_feature_construct()),reward,done

# EpiHEFT-CommGNN

当前提供正式主模型的预训练权重及 Pegasus-S 主实验测评，并保留 HEFT_Comm、PEFT、CPOP、IPPTS-Dyn 基线。其他代码仍在整理中。

## 安装环境

推荐 Linux、Python 3.12。已验证环境为 Python 3.12.3、PyTorch 2.8.0、PyG 2.5.3。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU推理
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

如使用CUDA，可将上面的PyTorch安装命令改为：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 使用权重推理

解压后进入本目录，执行：

```bash
# 一个完整调度实例：30个工作流，主模型与HEFT_Comm对比
python run.py --methods main,HEFT_Comm --H 0 --n 1 --workers 2

# 正式主实验：30个留出实例，H=0/3/5/10
python run.py --methods main,HEFT_Comm --H 0,3,5,10 --n 30 --workers 8

# 同时测评其余传统基线
python run.py --methods main,HEFT_Comm,PEFT,CPOP,IPPTS_Dyn --H 0,3,5,10 --n 30 --workers 8
```

默认使用随包 `main_actor.pth`。指定权重和GPU推理：

```bash
python run.py --weights main_actor.pth --device cuda:0 --methods main --H 0 --n 1
```

固定设置为正式seed0主模型、α=1.5、p_w=0.10、N=10、Pegasus-S、每实例30工作流、到达率5.4/h。测试索引30–59；`--n`取其前n个。代码自动在内存中重建工作流索引及到达时间，工作流XML已包含，无需下载数据或准备IL/cache。所有实例均运行至工作流全部完成。

每进程1个CPU线程；`--workers`控制并发。默认CPU推理。终端输出各方法FCT均值、实例SD(ddof=0)与相对HEFT_Comm的Gap；逐实例和汇总CSV写入新建的 `results/<时间>/`，也可用 `--output`指定空目录。单实例命令仅用于检查运行。

`model.py`为actor前向推理，`settings.py`仅包含固定推理参数，`env/`为共同调度与通信环境。本包不包含训练程序、训练配置或历史测评结果。

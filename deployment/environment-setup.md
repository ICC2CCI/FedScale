# 环境搭建

## 1. 系统要求

- Linux（Ubuntu 22.04+ 推荐）
- CUDA 12.1+（推荐 12.8）
- NVIDIA driver 535+

验证：

```bash
nvidia-smi
# 确认 GPU 型号、显存、driver 版本、CUDA 版本
```

## 2. 安装 Conda

如已有 conda 可跳过。

```bash
# Miniconda 安装
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
source ~/.bashrc
```

## 3. 创建 Python 环境

```bash
conda create -n flwr-ft python=3.10 -y
conda activate flwr-ft
```

## 4. 安装 PyTorch

```bash
# CUDA 12.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1（如果 driver 较旧）
# pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 应输出: 2.8.0+cu128 True NVIDIA H100 80GB HBM3（或其他 GPU）
```

## 5. 安装其他依赖

```bash
pip install transformers==4.45.2
pip install trl==0.8.6
pip install datasets==2.21.0
pip install accelerate==0.34.2
pip install flwr==1.18.0
pip install flwr-sim==1.18.0
pip install matplotlib numpy pandas
pip install modelscope==1.37.1
```

## 6. 验证安装

```bash
python -c "
import torch
import transformers
import trl
import flwr
import datasets
import modelscope

print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}')
print(f'flwr: {flwr.__version__}')
print(f'datasets: {datasets.__version__}')
print(f'modelscope: {modelscope.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

预期输出：

```
torch: 2.8.0+cu128
transformers: 4.45.2
trl: 0.8.6
flwr: 1.18.0
datasets: 2.21.0
modelscope: 1.37.1
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
```

## 7. 完整 pip install（一键安装）

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.45.2 trl==0.8.6 datasets==2.21.0 accelerate==0.34.2
pip install flwr==1.18.0 flwr-sim==1.18.0
pip install matplotlib numpy pandas modelscope==1.37.1
```

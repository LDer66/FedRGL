# FedRGL  
torch==1.12.1
torch-cluster==1.6.0
torch-scatter==2.1.0
torch-sparse==0.6.15
torch-spline-conv==1.2.1
torch-geometric==2.3.1

pip install torch==1.12.1 torch-cluster==1.6.0 torch-scatter==2.1.0 \
torch-sparse==0.6.15 torch-spline-conv==1.2.1 torch-geometric==2.3.1

python train_FedRGL_Cora.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3

python train_FedRGL_Cora_Ten.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3  

FedRGL/
│
├── train_FedRGL_Cora.py        # 训练脚本（每轮筛选噪声节点）
├── train_FedRGL_Cora_Ten.py    # 训练脚本（每 10 轮筛选噪声节点）
├── requirements.txt            # 依赖项列表
└── README.md                   # 项目说明文件



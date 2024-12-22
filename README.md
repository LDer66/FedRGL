
# FedRGL  

！[Correct the validation of different client numbers in the experiment section, i.e., Figure 5 of the experiment.](clients_number.png)  

**Figure:** Overview of the FedRGL framework. The architecture shows how data and models interact during training across clients and the server.

## 1. Requirements  
To run the project, ensure the following dependencies are installed:

```bash
torch==1.12.1  
torch-cluster==1.6.0  
torch-scatter==2.1.0  
torch-sparse==0.6.15  
torch-spline-conv==1.2.1  
torch-geometric==2.3.1  
```

Install the dependencies using pip:

```bash
pip install torch==1.12.1 torch-cluster==1.6.0 torch-scatter==2.1.0 \
torch-sparse==0.6.15 torch-spline-conv==1.2.1 torch-geometric==2.3.1
```

---

## 2. Usage  
### Train the Model  
To run the model on the **Cora** dataset with **5 clients**, use the following command:

```bash
python train_FedRGL_Cora.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3
```

---

## 3. Filter Frequency  
To perform noisy node filtering **every 10 rounds** instead of every round, use the following command:

```bash
python train_FedRGL_Cora_Ten.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3
```

---

## 4. Project Structure  
The project structure is organized as follows:

```plaintext
FedRGL/
│
├── train_FedRGL_Cora.py         # Training script (filter noisy nodes every round)
├── train_FedRGL_Cora_Ten.py     # Training script (filter noisy nodes every 10 rounds)
├── requirements.txt             # Dependency list
└── README.md                    # Project description file
```

---

##




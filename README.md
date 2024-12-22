
# FedRGL  

<span style="color:red; font-weight:bold; font-size:18px;">Image Correction</span>  

**Dear** chief editor and reviewer, due to our negligence, there are wrong pictures used in Figure 5 of the experimental part of the manuscript submitted. We sincerely apologize for our mistake and your misunderstanding of this part. The correct experimental diagram of Figure 5 is as follows:

![Correct the validation of different client numbers in the experiment section, i.e., Figure 5 of the experiment.](clients_number.png)  

**Figure:** Comparison of Prediction Accuracy Between FedRGL and Baselines on CiteSeer and Photo Datasets with Noise Rate of 0.3 Under Two Noise Types and Different Numbers of Clients.

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




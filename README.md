# SpikingResformer Face-Recognition 🚀

> A face-auth system using SpikingResformer backbone + ArcFace loss. Then I built flask backend for the model and next.js frontend. Enjoy!!

---

## 📋 Table of Contents
1. [Description](#description)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Prerequisites](#prerequisites)  
5. [Installation and usage](#installation)     
6. [Benchmarks](#benchmarks)  
7. [License](#license)  

---

## 📖 Description
This project leverages a **SpikingResformer** (an SNN-style Transformer) fine-tuned with **ArcFace** loss for face verification and identification.  
- **Backend**: FastAPI (Flask-style endpoints for `/register` & `/compare`)  
- **Frontend**: Next.js with on-screen webcam capture for user flows  
- **Storage**: SQLite for face embeddings  

> Why SpikingResformer + ArcFace?  
> - **SpikingResformer** for energy-efficient inference in edge scenarios.  
> - **ArcFace** loss to maximize inter-class margin → top accuracy on LFW (~0.90).  

---

## ✨ Features
- 🔥 **SNN Transformer** backbone (SpikingResformer)  
- 🎯 **ArcFace** loss for robust face embeddings  
- 📸 Webcam-based register & login flows in Next.js  
- 🔌 RESTful API with FastAPI (register, compare, health check)  
- 🐳 Docker support for dev & prod  

---

## 🛠️ Tech Stack
- **Python** 3.10+  
- **PyTorch** & **timm**  
- **facenet-pytorch** (MTCNN)  
- **FastAPI** / **Uvicorn**  
- **Next.js** (React) + Tailwind CSS  
- **SQLite**  
- **Docker** & **docker-compose**  

---

## ⚙️ Prerequisites
- **Git**  
- **Docker** & **Docker Compose**  
- **Node.js** 16+ (for frontend dev, if running outside container)  
- **Python** 3.10+ (only if you skip Docker)  

---

## 🚀 Installation and Usage

1. **Clone repo**  
   ```bash
   git clone https://github.com/your-username/spikingresformer-face.git
   cd spikingresformer-face

2. **Create logs folder inside backend folder**
    Then download this checkpoint for the model to run.
    https://drive.google.com/drive/folders/1rSvbhEJrYvU-omaj4BCAoe83wV_ZRIfI?usp=sharing

3. **Run docker command**
    Run this command: docker-compose up --build
    Then it will automatically build and use just need to test the demo on localhost:3000.


## 📈 Benchmarks

### 📁 Dataset
Evaluated on the [LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) benchmark.

- ✅ Test size: 13,000+ face pairs
- ✅ Preprocessed to 112x112 using MTCNN

### 🎯 Accuracy
| Metric        | Score                     |
|---------------|---------------------------|
| Accuracy      | 0.9042 (cosine similarity)|
| Threshold     |  0.1151                   |
| Embedding dim | 512                       |
| Time steps (T)| 4                         |

> 🧠 Uses ArcFace loss to boost inter-class margin and improve verification.


## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

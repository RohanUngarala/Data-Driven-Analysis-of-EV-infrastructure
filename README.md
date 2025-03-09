# 🚗 Velocity: Data-Driven Analysis of EV Charging Infrastructure for a Smart City 

![image](https://github.com/user-attachments/assets/525f3885-4f47-4a59-9491-97860a32d486)

### Medium/Heavy-Duty Vehicles

This project presents a **data-driven approach** to optimizing **Electric Vehicle (EV) charging infrastructure**, particularly for **medium and heavy-duty vehicles** within a **smart city ecosystem**. The research leverages **machine learning models, forecasting techniques, and optimization algorithms** to predict EV demand, optimize charging station locations, and ensure efficient energy resource allocation.

## Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Demo](#-demo)
- [Setup & Installation](#-setup--installation)
- [Contributors](#-contributors)

---

## 🔍 Overview
The **increasing adoption of EVs** requires a robust **charging infrastructure** that can cater to different vehicle types. This project addresses the **challenges of optimal charging station placement**, **energy demand forecasting**, and **vehicle range prediction** for **medium and heavy-duty vehicles** within a **smart city ecosystem**.

### 🏆 Objectives:
- 📈 **Forecast EV demand & charging station growth** using **time-series models** (Prophet).
- 🔋 **Predict vehicle range** to optimize charging efficiency using **Stacking Ensemble Regressor**.
- ⚡ **Anticipate energy demand** using a **Temporal Fusion Transformer** model.
- 📍 **Optimize new charging station locations** with **Linear Programming & K-means clustering**.

---

## ✨ Features
**EV Demand Forecasting** – Predict the number of heavy-duty EVs & required charging stations.  
**Range Prediction** – Estimate the driving range of EVs based on real-time charging data.  
**Energy Demand Forecasting** – Optimize **short & long-term** charging demand at stations.  
**Charging Station Placement Optimization** – Find the **best locations** for new stations.  
**Real-time Visualization** – Interactive **dashboards** for data insights & analysis.  

---

## System Architecture
Below is a high-level **system architecture** diagram illustrating the **data flow, models, and outputs**.

```
                     ┌───────────────────────────────┐
                     │        Data Sources           │
                     │ (EV Data, Charging Stats,     │
                     │  Geographic & Traffic Data)  │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌───────────────────────────────┐
                     │       Data Processing         │
                     │ (Preprocessing, Cleaning,     │
                     │  Feature Engineering)        │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
     ┌───────────────┬──────────────┬───────────────┐
     │ EV Demand    │ Range        │ Energy        │
     │ Forecasting  │ Prediction   │ Prediction    │
     │ (Prophet)    │ (Stacking)   │ (Transformer) │
     └───────────────┴──────────────┴───────────────┘
                                    │
                                    ▼
                     ┌───────────────────────────────┐
                     │   Charging Station Placement  │
                     │ (K-means + Linear Programming)│
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌───────────────────────────────┐
                     │        Visualization          │
                     │ (Dashboards, Reports, APIs)  │
                     └───────────────────────────────┘
```
---
![image](https://github.com/user-attachments/assets/5b5c58da-22df-4722-80cf-ccab9f2903ad)

Web Development:

Built on Flask Architecture for a user-friendly website.
HTML, CSS, and JavaScript for dynamic web pages.
AWS S3 for data storage and retrieval.

Database Management:

AWS S3 for secure and scalable storage of relevant EV charging-related data and predictions.

Machine Learning Models:

Leveraged tools such as Pandas, scikit-learn, TensorFlow, Prophet, and PuLP.
Collaborative platforms include Google Colab and Google Cloud.


## 🛠️ Technologies Used
- **Python** – Data processing, ML models  
- **Pandas, NumPy, Scikit-Learn** – Data analytics  
- **Prophet** – Time series forecasting  
- **TensorFlow, PyTorch** – Deep learning models  
- **PuLP** – Optimization for station placement  
- **Tableau / Power BI** – Data visualization  
- **AWS / Google Cloud** – Cloud-based infrastructure  

---

## 🎥 Demo
📺 **Watch a live demo of the project here:**  
[![Demo Video](https://img.youtube.com/vi/your-video-id/maxresdefault.jpg)](https://www.youtube.com/watch?v=your-video-id) *(Replace with actual video link)*  

💻 **Live Dashboard:** [EV Charging Analytics](https://your-dashboard-link.com) *(Replace with actual dashboard URL)*  

---

## Setup & Installation
To run the project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ev-charging-analysis.git
   cd ev-charging-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data pipeline**
   ```bash
   python data_processing.py
   ```

4. **Train the models**
   ```bash
   python train_models.py
   ```

5. **Start the visualization dashboard**
   ```bash
   streamlit run dashboard.py
   ```

---

**Project Supervisor:** [Dr. Jerry Gao](https://www.sjsu.edu/people/jerry.gao/)  

---


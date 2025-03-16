# 🚗 Velocity: Data-Driven Analysis of EV Charging Infrastructure for a Smart City 

![image](https://github.com/user-attachments/assets/525f3885-4f47-4a59-9491-97860a32d486)

### Revolutionizing Electric Vehicle (EV) Infrastructure Planning - VELOCITY

This Velocity project presents a **data-driven approach** to optimizing **Electric Vehicle (EV) charging infrastructure**, particularly for **medium and heavy-duty vehicles** within a **smart city ecosystem**. The research leverages **machine learning models, forecasting techniques, and optimization algorithms** to predict EV demand, optimize charging station locations, and ensure efficient energy resource allocation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Demo](#demo)
- [Setup & Installation](#setup--installation)

---

## Overview
The **increasing adoption of EVs** requires a robust **charging infrastructure** that can cater to different vehicle types. This project addresses the **challenges of optimal charging station placement**, **energy demand forecasting**, and **vehicle range prediction** for **medium and heavy-duty vehicles** within a **smart city ecosystem**.

### Objectives:
- **Forecast EV demand & charging station growth** using **time-series models** (Prophet Model).
- **Predict vehicle range** to optimize charging efficiency using **Stacking Ensemble Regressor**.
- **Anticipate energy demand** using a **Temporal Fusion Transformer** model.
- **Optimize new charging station locations** with **Linear Programming & K-means clustering**.

---

## Features
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
                     │ (Dashboards, Reports, APIs)   │
                     └───────────────────────────────┘
```
---
![image](https://github.com/user-attachments/assets/5b5c58da-22df-4722-80cf-ccab9f2903ad)

## Web Development:

- Built on Flask Architecture for a user-friendly website.
- HTML, CSS, and JavaScript for dynamic web pages.
- AWS S3 for data storage and retrieval.

## Database Management:

- AWS S3 for secure and scalable storage of relevant EV charging-related data and predictions.

## Machine Learning Models:

- Leveraged tools such as Pandas, scikit-learn, TensorFlow, Prophet, and PuLP.
- Collaborative platforms include Google Colab and Google Cloud.


## Technologies Used
- **Python** – Data processing, ML models  
- **Pandas, NumPy, Scikit-Learn** – Data analytics  
- **Prophet** – Time series forecasting  
- **TensorFlow, PyTorch** – Deep learning models  
- **PuLP** – Optimization for station placement  
- **Tableau / Power BI** – Data visualization  
- **AWS / Google Cloud** – Cloud-based infrastructure  

---

## Demo

## Home Page of the Velocity:
![image](https://github.com/user-attachments/assets/514100c0-2e55-461c-b1f1-52095544835f)

- Overview of project divisions and functionalities.

## Charging Census Page:
- **Shows projected EV counts and the necessary charging stations by county and zip code.**  
- **Allows drill-down to view the count of each vehicle type and the locations of charging infrastructure.**

![image](https://github.com/user-attachments/assets/23f795d3-fa7e-43ed-a534-e0d8727d2c28)

<img width="853" alt="image" src="https://github.com/user-attachments/assets/814a2726-af55-4702-9e0a-b4715e73b053" />

## Charging Snapshot Page:
- **Displays energy usage requirements for each city based on user input.**
- **Provides graphs illustrating load profiles for each hour of the day ( both in weekday and weekend).**

![image](https://github.com/user-attachments/assets/f37c0b39-08dd-40b1-bfb7-56b79e74e38b)

![image](https://github.com/user-attachments/assets/c40b1a32-f943-418f-acea-9efd40dfa6a7)

## Charge Map Page:
- **Shows existing and proposed charging stations based on zip code and vehicle type.** 
- **Provides details on station names, addresses, connector types, and projected future requirements.**

![image](https://github.com/user-attachments/assets/4561fda3-2190-4391-a112-26bacbddcf39)

---
## Setup & Installation

To run the project locally, follow these steps:

## Setup & Installation

To run the project locally, follow these steps:

1. **Clone the repository**  
   Open a terminal and run:
   ```bash
   git clone https://github.com/your-username/velocity.git
   cd velocity
2. **Install dependencies**
Install the required packages using pip:
pip install -r requirements.txt
3. **Run the Flask application**
Start the Flask server by running:
flask run
4. **Train the models**
Execute the following command to train models:
python train_models.py
5. **Access the website**
Open your browser and visit:
http://localhost:5000/
6. **Access the visualization dashboard (https://public.tableau.com/app/profile/lohitha.vanteru6992/viz/MarketAnalysisofHDMD-EVCSI_17043325836300/Story2)**
---
**Project Supervisor:** [Dr. Jerry Gao](https://www.sjsu.edu/people/jerry.gao/)  
---
## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

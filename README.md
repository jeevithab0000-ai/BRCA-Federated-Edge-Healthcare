# BRCA-Federated-Edge-Healthcare
Privacy-preserving breast cancer prediction using Federated Learning and Edge Computing

## Overview

This project presents a privacy-preserving healthcare prediction system that combines Federated Learning (FL) and Edge Computing to support breast cancer prediction across multiple simulated hospital environments.

Instead of collecting medical data from all hospitals into one centralized location, each hospital trains its machine learning model locally. Only model parameters are exchanged with the federated server, which aggregates the participating models to produce an improved global model.

The trained global model is deployed at an edge device through a Flask API, enabling predictions through a web-based healthcare dashboard.

The project demonstrates how distributed machine learning, edge deployment, and privacy-conscious data processing can be integrated into a healthcare application.


## Problem Statement

Traditional centralized machine learning systems require medical datasets from multiple healthcare institutions to be collected and stored in a common location. Sharing sensitive medical data creates privacy, security, and data governance challenges.

Healthcare institutions may also have limitations regarding data sharing, connectivity, and centralized infrastructure.

This project addresses these challenges by implementing a federated learning approach in which hospitals train locally while sharing model updates rather than raw patient records.


## Proposed Solution

The proposed system consists of four simulated hospital clients and a federated server.

 Each hospital maintains its own local dataset.
 Local machine learning training takes place within each hospital's environment.
 Model parameters are shared with the federated server.
 The server aggregates client updates using Federated Averaging (FedAvg).
 A global model is generated through iterative training rounds.
 The trained model is deployed to an edge environment.
 A Flask-based API provides prediction services to the frontend dashboard.

## System Architecture

mermaid
flowchart TD
    A[Healthcare Dashboard - React] --> B[Flask Edge API]
    B --> C[Edge Device - Global Model]
    C --> D[Breast Cancer Prediction]

    E[Hospital A - Local Data] --> F[Federated Training]
    G[Hospital B - Local Data] --> F
    H[Hospital C - Local Data] --> F
    I[Hospital D - Local Data] --> F

    F --> J[Federated Server - FedAvg]
    J --> F
    J --> K[Global Model]
    K --> C

### Architecture Components

1. Hospital Clients

Four simulated hospitals (Hospital A, B, C, and D) hold separate local datasets. Each client trains the model using its own data without directly sharing patient records.

2. Federated Learning Server

The central federated server coordinates training rounds, receives model parameters from participating hospitals, and aggregates the updates using the Federated Averaging algorithm.

3. Federated Averaging (FedAvg)

FedAvg combines model weights from multiple participating clients to create a global model. Client model updates are aggregated based on the federated training configuration and client data sizes.

4. Edge Computing

The trained global model is deployed in an edge environment to support prediction without requiring every inference request to go through the federated training server.

5. Flask API

The Flask backend provides endpoints for health checks and prediction requests, allowing the frontend to communicate with the edge model.

6. Healthcare Dashboard

The React frontend provides a user interface for predictions, monitoring hospitals and edge devices, viewing federated learning status, and displaying analytics.

## Key Features

 Privacy-conscious distributed model training
 Four simulated hospital clients
 Federated Learning using Flower
 Federated Averaging (FedAvg)
 Local training on hospital datasets
 Global model aggregation
 Edge-based machine learning inference
 Flask-based REST API
 React healthcare dashboard
 Breast cancer prediction interface
 Hospital/client status monitoring
 Federated training status visualization
 Model performance analytics
 Prediction results and confidence display
 Separation between training and inference components
## Technologies Used

 Category | Technologies 
 Programming Language | Python 
 Machine Learning | Scikit-learn 
 Federated Learning | Flower (FL) 
 Model Algorithm | Machine Learning classifier 
 Data Processing | Pandas, NumPy 
 Backend | Flask 
 API | REST API 
 Frontend | React.js, JavaScript, HTML, CSS 
 Frontend Tooling | Vite 
 Model Persistence | Pickle 
 Development Tools | PyCharm, VS Code 
 Dataset | Wisconsin Diagnostic Breast Cancer (WDBC) 

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, a publicly available dataset containing numerical features derived from breast mass measurements.

The dataset includes features associated with digitized images of fine needle aspirates (FNA) of breast masses.

### Dataset Characteristics

 Breast cancer diagnostic data
 Numerical feature representation
 Benign and malignant diagnostic categories
 Feature-based machine learning classification
 Data distributed among simulated hospital clients

The data is divided into separate hospital datasets to simulate decentralized healthcare environments.

> This project is an academic machine learning prototype. The dataset is not a live hospital database, and the system is not a medical diagnostic tool.

## Federated Learning Workflow

The federated training process follows these steps:

1. Prepare and preprocess the breast cancer dataset.
2. Divide the dataset into four hospital-specific datasets.
3. Initialize the federated learning server.
4. Distribute the initial global model parameters to participating clients.
5. Each hospital trains its model locally.
6. Hospital clients return model updates to the server.
7. The server aggregates the updates using FedAvg.
8. Update the global model.
9. Repeat the process over multiple training rounds.
10. Save the final global model for edge deployment.

### Federated Training Configuration

 Parameter | Configuration 
 Federated framework | Flower 
 Aggregation algorithm | FedAvg 
 Hospital clients | 4 
 Training rounds | 10 
 Client participation | All configured clients 
 Local data | Hospital-specific datasets 
 Server address | 127.0.0.1:8080 (local development) 

## Edge Computing and Prediction Workflow

After federated training is completed, the global model is saved and deployed to the edge application.

The edge prediction workflow:

1. A user opens the healthcare dashboard.
2. The user enters patient feature values into the prediction form.
3. The frontend sends the input to the Flask backend.
4. The backend preprocesses the input using the saved scaler.
5. The global machine learning model generates a prediction.
6. The API returns the prediction result.
7. The dashboard displays the result.

The edge environment is designed to provide local inference using the deployed model without needing to perform federated training for each prediction request.

## Privacy and Security

The project demonstrates a privacy-conscious Federated Learning architecture.

 Hospital datasets remain separated during federated training.
 Raw training records are not intentionally sent to the federated aggregation server.
 Clients share model parameters for aggregation.
 The global model is used for edge inference.
 The system illustrates decentralized training concepts.

Important: Federated Learning alone does not guarantee complete privacy. Model updates may still reveal information. Production healthcare systems would require additional safeguards such as secure aggregation, differential privacy, access control, encryption, auditing, and compliance with applicable healthcare regulations.

## Project Structure

text
BRCA_Federated_Edge_Project/
│
├── app.py
├── federated_client.py
├── split_hospitals.py
│
├── federated_server/
│   └── server.py
│
├── models/
│   ├── scaler.pkl
│   └── global_model.pkl
│
├── hospitals/
│   ├── hospital_A/
│   │   ├── data.csv
│   │   └── edge/
│   │       └── edge_A.py
│   ├── hospital_B/
│   │   ├── data.csv
│   │   └── edge/
│   │       └── edge_B.py
│   ├── hospital_C/
│   │   ├── data.csv
│   │   └── edge/
│   │       └── edge_C.py
│   └── hospital_D/
│       ├── data.csv
│       └── edge/
│           └── edge_D.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   
│
└── README.md

*The folder structure above represents the core project organization. Additional files may exist in the actual implementation.*

## Software Requirements

 Python 3.10
 Node.js and npm
 PyCharm or VS Code
 Git
 Web browser
 Windows, Linux, or macOS


## Running the Project

The federated training server, hospital clients, backend API, and frontend are separate components. Start each component using the appropriate script and configuration for your project.

### Federated Training

Run the federated server and connect the configured hospital clients to initiate local training and global aggregation.

### Backend API

From the project root, start the Flask application:

bash
python app.py

The backend API is intended to run locally on the configured port.

### Frontend Dashboard

From the frontend directory, run:

bash
npm run dev

Open the local Vite URL shown in your terminal to access the dashboard.

> Note: The exact start order and commands may depend on the Flower version and the scripts currently implemented in your repository.


## Application Modules

### Dashboard

Provides an overview of the healthcare federated system, including model and deployment status.

### Breast Cancer Prediction

Accepts patient feature values and returns a prediction from the deployed global model.

### Hospitals

Displays the simulated hospital clients participating in the distributed learning environment.

### Edge Devices

Represents the edge inference environments hosting the deployed global model.

### Federated Learning

Displays information about federated training, client participation, and model aggregation.

### Analytics

Displays model performance and system-level visualizations.

### Privacy and Security

Explains the decentralized training approach and the privacy-oriented design of the application.

## Model Evaluation

The project can evaluate model performance using metrics such as:

 Accuracy
 Precision
 Recall
 F1-score
 Confusion matrix

## Advantages

- Reduces the need to centralize training datasets
- Supports decentralized learning across multiple clients
- Simulates collaboration among healthcare institutions
- Combines machine learning with edge deployment
- Supports a modular architecture
- Separates model training and prediction services
- Demonstrates a privacy-oriented approach to healthcare AI

## Limitations

- Uses a public benchmark dataset rather than real hospital records.
- Hospitals are simulated clients.
- The application is an academic prototype.
- Federated training does not automatically provide formal privacy guarantees.
- Clinical validation has not been established.
- Model performance depends on dataset quality, preprocessing, and training configuration.
- The system must not be used to make real-world medical decisions.

## Future Enhancements

- Secure Aggregation
- Differential Privacy
- Real-time hospital client monitoring
- Improved edge deployment
- Docker-based deployment
- Cloud integration
- Authentication and role-based access control
- Model drift monitoring
- Advanced model evaluation
- Explainable AI using SHAP or other interpretability techniques
- Integration with securely managed clinical data systems after appropriate validation and authorization

### Domain

Machine Learning | Federated Learning | Edge Computing | Healthcare AI | Privacy-Preserving Systems


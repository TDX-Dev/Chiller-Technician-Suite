# TAM - HACK PROJECT [ CHILLER DIAGNOSIS FOR TECHNICIANS ] [ BY TEAM: THE DEVOID ]

This project implements neural networks using PyTorch for predicting data based on sensor inputs, along with a web interface for interacting with the model and visualizing data trends. The PyTorch model is converted into the ONNX format for use with ONNX.js to enable inference directly in the browser. In addition to neural networks, Linear Regression models are used for pump frequency and chiller load percentage predictions, while an LSTM model is employed for time series forecasting.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the Web Server](#running-the-web-server)
- [Endpoints](#endpoints)
- [How to Use](#how-to-use)
- [Model Conversion](#model-conversion)
- [Credits](#credits)

## Project Overview
This project combines neural networks trained with PyTorch and a web-based interface built with Node.js and ExpressJS. Users can upload sensor data to visualize trends or manually input data to get predictions for chiller load and pump frequency, which helps to optimize plant efficiency. The project also uses an LSTM model for time series prediction and Linear Regression for pump frequency and chiller load percentage predictions.

## Features
1. **Data Upload & Visualization**: Upload CSV files containing sensor data and visualize trends using Chart.js.
2. **Manual Input for Chiller Load Calculation**: Input parameters manually to calculate chiller load percentage using a Linear Regression model.
3. **Manual Input for Pump Frequency Calculation**: Input parameters manually to compute pump frequency for ideal efficiency using a Linear Regression model.
4. **Time Series Prediction**: The LSTM model is used for forecasting based on time series data.
5. **PyTorch Model Integration**: The backend neural network model, originally built in PyTorch, is converted to ONNX format and executed using ONNX.js in the browser.

## Tech Stack
- **Backend**: Node.js, ExpressJS
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: PyTorch, ONNX, ONNX.js, Linear Regression, LSTM
- **Data Visualization**: Chart.js

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/project-name.git
    cd project-name
    ```

2. Install dependencies:

    ```bash
    npm install
    ```

3. Convert the PyTorch model to ONNX format. Follow the instructions under [Model Conversion](#model-conversion) to export the PyTorch model to `model.onnx`.

## Running the Web Server
To launch the web server and access the application:

1. Start the server:
    ```bash
    node .
    ```

2. Open your browser and visit:
    ```
    http://localhost:8080/data/
    ```

### Additional Endpoints:
- **Login page**:  
    `http://localhost:8080/login/`

## Endpoints
- `/data/`: Main endpoint for uploading sensor data and visualizing the trends.
- `/login/`: Optional login page (can be customized or expanded based on your needs).

## How to Use
1. **Upload Sensor CSV**: Navigate to `http://localhost:8080/data/`, and upload your sensor data in CSV format. The server processes the data and visualizes trends using Chart.js.
2. **Chiller Load Prediction**: Input data manually into the provided form to get predictions for chiller load percentage using a Linear Regression model.
3. **Pump Frequency Prediction**: Input data manually into the provided form to get pump frequency recommendations for ideal efficiency using a Linear Regression model.
4. **Time Series Forecasting**: The LSTM model is used to predict future sensor data based on previous time series data.

## Model Conversion
The PyTorch neural network model was exported to ONNX format for compatibility with ONNX.js, enabling model inference in the browser. To convert the model, follow these steps:

1. Install the necessary libraries:

    ```bash
    pip install torch onnx
    ```

2. Convert the PyTorch model to ONNX:

    ```python
    import torch
    import onnx
    from your_model import YourModel  # Replace with your model

    model = YourModel()
    model.load_state_dict(torch.load('model.pth'))  # Replace with your model checkpoint

    # Dummy input for the model (adjust size to match your model's input)
    dummy_input = torch.randn(1, 15)  # Example input tensor
    torch.onnx.export(model, dummy_input, "model.onnx")
    ```

3. Place the `model.onnx` file in the `/web/models/` directory to be used by the web interface.

## Credits
- **ONNX.js**: For running neural network inference in the browser.
- **Chart.js**: For visualizing data trends.

---

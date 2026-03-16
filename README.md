# 🌱 Smart Agri Tech Project

**Smart Agriculture Solution for Crop Recommendation, Fertilizer Prediction, and Crop Yield Forecasting**

An intelligent farming assistant that leverages machine learning to help farmers make data-driven decisions. The system provides crop recommendations based on soil parameters, suggests optimal fertilizers, and predicts crop yields to maximize agricultural productivity.

## ✨ Features

- **Crop Recommendation** - Suggests the best crops to cultivate based on soil nutrients, pH, and climatic conditions
- **Fertilizer Prediction** - Recommends appropriate fertilizers based on soil deficiencies and crop requirements  
- **Crop Yield Prediction** - Forecasts expected crop yield using historical data and current parameters
- **User-friendly Interface** - Clean and intuitive React-based UI for easy interaction

## 🛠️ Tech Stack

### Frontend
- React.js (v20.19.6)
- npm (v10.9.1)
- HTML5/CSS3
- JavaScript (ES6+)

### Backend
- Python (v3.12.0)
- Flask/FastAPI
- pip (v26.0.1)

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib (for model persistence)

## 📋 Prerequisites Check (Windows)

Run these commands in **PowerShell** or **Command Prompt** to verify your installation:

```powershell
# Check Python version
python --version
# Expected: Python 3.12.0

# Check pip version
pip --version  
# Expected: pip 26.0.1

# Check Node version
node -v
# Expected: v20.19.6

# Check npm version
npm -v
# Expected: 10.9.1
```

### If pip needs update:
```powershell
python -m pip install --upgrade pip
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/UdhyaKumarKMIT/Smart-Agri-Tech.git
cd Smart-Agri-Tech
```

### 2. Backend Setup (Python)

Navigate to backend directory and install dependencies:
```powershell
cd agri-tech-backend

# Install required Python packages
pip install -r requirements.txt

# Start the backend server
python app.py
```
The backend server will start at `http://localhost:5000`

### 3. Frontend Setup (React)

Open a **new terminal** and navigate to the frontend directory:
```powershell
cd agri-tech-ui

# Install npm dependencies
npm install

# Start the React development server
npm run dev
```
The frontend application will open at `http://localhost:3000`

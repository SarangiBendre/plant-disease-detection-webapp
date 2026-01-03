# 🌿 Plant Disease Detection Web Application

A deep learning–based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). Users can upload a plant leaf image through a web interface and receive the predicted disease along with a confidence score.

---

## 🚀 Live Demo
🔗 https://plant-disease-detection-webapp-1i17.onrender.com

> Note: On the free hosting tier, the first request may take a few seconds to load.

---

## 📌 Project Overview

Plant diseases can significantly reduce agricultural productivity. This project aims to assist farmers and researchers by providing an automated system to identify plant diseases using deep learning and image processing techniques.

The application is deployed as a web-based system, allowing easy access without requiring any hardware setup.

---

## 🧠 Features

- Upload plant leaf images
- Automatic disease detection using CNN
- Displays predicted disease name
- Shows prediction confidence score
- Simple and user-friendly web interface
- Deployed on the cloud for public access

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Convolutional Neural Network (CNN)  
- Flask  
- HTML & CSS  
- Render (Cloud Deployment)

---

## 📂 Project Structure

```

plant-disease-detection-webapp/
│
├── app.py
├── plant_disease_model.keras
├── class_names.json
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
└── README.md

````

---

## 📊 Dataset

- PlantVillage Dataset
- Contains images of healthy and diseased plant leaves
- Used to train the CNN classification model

---

## ⚙️ How the System Works

1. User uploads a plant leaf image.
2. The image is resized and normalized.
3. The CNN model predicts the disease class.
4. The predicted disease and confidence score are displayed on the webpage.

---

## ▶️ Run the Project Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/plant-disease-detection-webapp.git
cd plant-disease-detection-webapp
````

### Step 2: Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install required dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python app.py
```

Open the browser and go to:

```
http://127.0.0.1:5000
```

---

## 🎓 Academic & Interview Summary

Developed a plant disease detection system using a Convolutional Neural Network trained on the PlantVillage dataset and deployed it as a Flask web application for real-time disease prediction.

---

## 📌 Future Enhancements

* Add disease description and treatment suggestions
* Improve UI design and responsiveness
* Support multiple plant species
* Deploy using a production WSGI server (Gunicorn)

---

## 👨‍💻 Author

**Sarangi Bendre**
B.Tech – Artificial Intelligence & Machine Learning

---

## ⭐ Acknowledgements

* PlantVillage Dataset
* TensorFlow and Flask Documentation
* Render Cloud Platform

```

---

✅ This README is **professional, clean, and internship-ready**.  
If you want, I can now help you write:
- Resume project points  
- LinkedIn post  
- Final project report PDF  

Just tell me 👍
```

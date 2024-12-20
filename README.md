# Alzheimer’s Disease Detection Web App

A web application for predicting stages of **Alzheimer’s disease** using a deep learning model. The app accepts an MRI image and outputs the predicted stage of Alzheimer’s disease.

---
![Web App Demo](/demo.png)
---

## Features

- Upload an **MRI image** in `.jpg` format.
- Predicts one of the following stages of Alzheimer’s disease:
   - **Non Demented**
   - **Very Mild Demented**
   - **Mild Demented**
   - **Moderate Demented**
- Displays the prediction result in a user-friendly interface.
- Provides **confidence scores** for each class.
- Generates a **Grad-CAM visualization** to explain predictions.

<!-- ---

## Deployed Application

The web application is deployed and accessible at:

**[http://<your-deployed-ip>:8000](http://<your-deployed-ip>:8000)**

Replace `<your-deployed-ip>` with the actual IP address or domain where the application is hosted. -->

---

## Requirements

- Python 3.8+
- Flask

---

## Setup and Installation

### Download Trained Models

Due to file size limitations, the `.h5` file for the Trained model is not included directly in the repository. Instead, it has been uploaded to Google Drive.

1. Download the Trained model from the following link:
   **[Download Trained Models](https://drive.google.com/file/d/1RbERyAQFur8sEjeuVhg4BZLumPCgvaZ2/view?usp=drive_link)**

2. Extract the zip file into the `models/` directory:
   ```bash
   unzip models.zip -d models/
   ```

Ensure that the extracted `.h5` file is located at `models/fine_tuned_vgg16.h5`.

---

### Linux

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and Extract Model:**
   Follow the steps in the **Download Trained Models** section.

5. **Run the Flask App:**
   ```bash
   flask run --host=0.0.0.0 --port=8000
   ```

6. **Access the Web App:**
   Open your browser and navigate to `http://127.0.0.1:8000/` or the deployed IP if hosting externally.

---

### Windows

1. **Clone the Repository:**
   ```powershell
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download and Extract Model:**
   Follow the steps in the **Download Trained Models** section.

5. **Run the Flask App:**
   ```powershell
   flask run --host=0.0.0.0 --port=8000
   ```

6. **Access the Web App:**
   Open your browser and navigate to `http://127.0.0.1:8000/` or the deployed IP if hosting externally.

---

## Project Structure

```plaintext
<repository-folder>/
|
|-- app.py             # Main Flask application
|-- templates/         # HTML templates
|-- static/            # Static files (CSS, JS, images)
|-- model/             # Trained models files
|-- notebooks/         # Jupyter notebooks used for model training
|-- requirements.txt   # Python dependencies
```
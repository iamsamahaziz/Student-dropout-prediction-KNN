# 🎓 Student Dropout Prediction using K-Nearest Neighbors (KNN)

A Machine Learning project that predicts student academic outcomes (Dropout, Enrolled, or Graduate) using the K-Nearest Neighbors algorithm implemented from scratch.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Implementation](#model-implementation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Author](#author)

## 🎯 Overview

This project aims to predict whether a student will:
- **Dropout** - Leave the program before completion
- **Enrolled** - Continue studying
- **Graduate** - Successfully complete their studies

Early identification of at-risk students can help educational institutions provide timely support and interventions.

## 📊 Dataset

The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (ID: 697).

### Dataset Characteristics
- **Source**: Portuguese higher education institutions
- **Instances**: 4,424 students
- **Features**: 36 attributes
- **Target**: 3 classes (Dropout, Enrolled, Graduate)

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | Marital status, Gender, Age at enrollment, Nationality |
| **Academic Background** | Previous qualification, Previous qualification grade, Admission grade |
| **Family Background** | Mother's/Father's qualification, Mother's/Father's occupation |
| **Enrollment Info** | Application mode, Application order, Course, Daytime/evening attendance |
| **Academic Performance** | Curricular units (1st & 2nd sem): credited, enrolled, evaluations, approved, grade |
| **Financial** | Tuition fees up to date, Debtor, Scholarship holder |
| **Socioeconomic** | Unemployment rate, Inflation rate, GDP |
| **Other** | Displaced, Educational special needs, International |

## ✨ Features

- ✅ **Custom KNN Implementation** - Algorithm built from scratch using NumPy
- ✅ **Data Preprocessing** - StandardScaler for feature normalization
- ✅ **Distance Calculation** - Euclidean distance metric
- ✅ **Majority Voting** - Classification based on k-nearest neighbors

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip

### Clone the Repository
```bash
git clone https://github.com/yourusername/student-dropout-prediction.git
cd student-dropout-prediction
```

### Install Dependencies
```bash
pip install pandas numpy seaborn matplotlib scikit-learn ucimlrepo
```

## 🚀 Usage

### Running the Jupyter Notebook
```bash
jupyter notebook KNN_students_data.ipynb
```

### Quick Start
```python
from ucimlrepo import fetch_ucirepo

# Load dataset
data = fetch_ucirepo(id=697)
X = data.data.features.values
y = data.data.targets.values

# Split and scale data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 🔧 Model Implementation

### Distance Function
```python
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
```

### K-Nearest Neighbors Function
```python
def kni(x_train, xi, k):
    distances = []
    for i in range(len(x_train)):
        d = distance(xi, x_train[i])
        distances.append([d, i])
    distances.sort()
    distances = np.array(distances[:k])
    return distances[:, -1]
```

### Prediction
The model uses majority voting among the k-nearest neighbors to determine the class of a new sample.

## 📈 Results

The KNN classifier is evaluated using:
- **Accuracy Score** - Overall prediction accuracy
- **Train/Test Split** - 80% training, 20% testing

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Programming Language |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data Manipulation |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ML Utilities |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Data Visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) | Statistical Visualization |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive Development |

## 📁 Project Structure

```
KNN/
├── KNN_students_data.ipynb    # Main Jupyter notebook
├── data.csv                   # Dataset file
└── README.md                  # Project documentation
```

## 🔮 Future Improvements

- [ ] Hyperparameter tuning for optimal k value
- [ ] Cross-validation implementation
- [ ] Comparison with other classification algorithms
- [ ] Feature importance analysis
- [ ] Confusion matrix visualization
- [ ] ROC curve analysis

## 📚 References

- [UCI ML Repository - Student Dropout Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [K-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

## 👤 Author

**Samah Aziz**

- GitHub: [@iamsamahaziz](https://github.com/iamsamahaziz)

---

⭐ If you found this project helpful, please consider giving it a star!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

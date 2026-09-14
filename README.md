# Affective-Cognitive Digital Twins for Predictive Personalized Learning

## Project Overview

**Affective-Cognitive Digital Twins for Predictive Personalized Learning** is an AI-based personalized learning system that models learner behavior, predicts future learning states, simulates possible learning outcomes, and recommends suitable learning actions.

Traditional personalized learning systems mainly recommend the next activity based on past performance. This project aims to go beyond reactive recommendation by simulating possible learner futures and selecting actions that may lead to better learning outcomes.

The system uses learner interaction data, machine learning models, learner-state estimation, action transition modeling, and future-state simulation to provide personalized learning recommendations.

> **Important:** The current implementation uses behavioral and performance-based features to estimate learner states. Direct emotional measurements are not included in the selected dataset. Therefore, affective states are treated as proxy-estimated states rather than directly measured emotions.

---

## Objectives

The main objectives of this project are:

* Analyze learner interaction and performance data.
* Estimate the current learning state of a student.
* Predict the learner's future performance or state.
* Compare possible learning actions or interventions.
* Simulate possible future learner trajectories.
* Recommend personalized learning actions.
* Support proactive and adaptive learning decisions.
* Visualize model performance and learner-state transitions.

---

## Dataset

### Dataset Used

This project uses the:

**ASSISTments 2009–2010 Skill Builder Dataset**

The dataset contains learner interaction records related to educational activities and problem-solving behavior.

Typical information available in the dataset includes:

* Student or user ID
* Problem ID
* Skill or knowledge component
* Correctness of the response
* Number of attempts
* Hint-related information
* Interaction history
* Response-related features
* Timestamps or ordering information, depending on the dataset file

### Dataset Source

The dataset can be obtained from the official ASSISTments data page:

https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

### Dataset Setup

The raw dataset is not included in this repository because of its size.

After downloading the dataset, place the CSV file inside:

```text
data/raw/
```

Example project structure:

```text
data/
└── raw/
    └── skill_builder_data.csv
```

The exact filename may differ depending on the downloaded dataset file.

### Affective-Cognitive State Estimation

The selected ASSISTments 2009–2010 dataset does not directly provide physiological or emotional measurements such as boredom, frustration, confusion, or engagement.

Therefore, this project estimates learner affective-cognitive states using behavioral and performance-based proxy features, such as:

* Correctness trends
* Number of incorrect responses
* Attempts
* Hint usage
* Recent learning performance
* Historical interaction patterns
* Predicted learning outcomes

These features are used to identify learner states such as:

* Initial or unknown state
* High struggle
* Moderate struggle
* Stable learning
* Good or improving performance

---

## System Architecture

The overall workflow of the project is:

```text
Raw Learner Data
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
Learner-State Creation
        |
        v
Learner Trajectory Generation
        |
        v
Machine Learning Models
        |
        v
Action Transition Model
        |
        v
Future-State Simulation
        |
        v
Personalized Recommendation
        |
        v
Evaluation and Visualization
```

---

## Main Modules

### 1. Data Preprocessing

The raw learner interaction data is cleaned and prepared for machine learning.

This stage may include:

* Handling missing values
* Removing unnecessary columns
* Converting data types
* Sorting learner interactions
* Creating learner-level features
* Preparing training and testing data

---

### 2. Feature Engineering

Relevant features are extracted from learner interaction records.

Examples of engineered features include:

* Correctness-based features
* Attempt-based features
* Hint-related features
* Previous performance
* Recent performance
* Cumulative performance
* Learner interaction history
* Performance trends

These features help the system understand the learner's current condition and predict future outcomes.

---

### 3. Learner-State Creation

Learners are assigned an estimated learning state based on their behavioral and performance patterns.

Possible states include:

* **Initial State:** Not enough information is available about the learner.
* **High Struggle:** The learner shows poor performance or repeated difficulty.
* **Moderate Struggle:** The learner experiences some difficulty but may improve with support.
* **Stable Learning:** The learner is progressing at a relatively consistent level.
* **Good/Improving:** The learner demonstrates strong or improving performance.

These states are estimated from learning behavior and should not be interpreted as direct psychological or emotional measurements.

---

### 4. Learner Trajectory Generation

Learner interaction records are converted into sequences or trajectories.

A trajectory represents the learner's progression over time.

For example:

```text
Initial State → Moderate Struggle → Stable Learning → Good Performance
```

These sequences are used to train models that predict future learner states.

---

### 5. Machine Learning Models

The project experiments with multiple machine learning models.

The models include:

* Logistic Regression
* Random Forest
* Long Short-Term Memory Network
* Balanced LSTM model for handling class imbalance

The models are used to predict learner performance or future learner states.

#### Logistic Regression

Logistic Regression is used as a baseline classification model.

#### Random Forest

Random Forest is used to capture nonlinear relationships between learner features and predicted outcomes.

#### LSTM

The Long Short-Term Memory network is used to process sequential learner interaction data and identify patterns over time.

#### Balanced LSTM

The Balanced LSTM model is used to improve learning-state prediction when some learner states occur less frequently than others.

---

### 6. Action Transition Model

The action transition model represents how different learning actions may influence a learner's future state.

Possible learning actions include:

* Practice more questions
* Review a concept
* Receive additional explanation
* Attempt an easier problem
* Attempt a similar problem
* Continue with the current learning activity

The model estimates possible state changes after applying a learning action.

Example:

```text
High Struggle + Concept Review → Possible Stable Learning
Moderate Struggle + Extra Practice → Possible Improvement
Stable Learning + Advanced Problem → Possible Good Performance
```

The transition probabilities are estimates based on learner behavior and model predictions. They do not represent guaranteed outcomes.

---

### 7. Future-State Simulation

The system simulates possible learner futures by considering different learning actions.

For each possible action, the system estimates:

* The possible next learner state
* The expected learning outcome
* The estimated improvement
* The suitability of the action for the current learner

The action with the best estimated outcome can then be selected as the recommended action.

The simulation component supports the digital-twin concept by representing possible future learner states before an action is actually applied to the learner.

---

### 8. Personalized Recommendation

The system recommends learning actions based on the learner's estimated current state.

Example recommendations:

| Learner State     | Example Recommendation                          |
| ----------------- | ----------------------------------------------- |
| Initial State     | Start with diagnostic or introductory questions |
| High Struggle     | Review concepts and provide easier practice     |
| Moderate Struggle | Provide guided practice and explanations        |
| Stable Learning   | Continue practice with moderate difficulty      |
| Good/Improving    | Introduce advanced or challenging questions     |

The recommendation is intended to be personalized according to the learner's predicted needs.

---

## Digital Twin Concept

In this project, the digital twin is represented computationally through:

* The learner's current estimated state
* The learner's historical behavior
* The predictive machine learning models
* The action transition model
* The simulated future learner states
* The recommendation mechanism

The **action transition model alone is not the complete digital twin**. It is one important component of the digital-twin system.

The digital twin is the combination of the learner-state representation, prediction models, transition model, future simulation, and personalized decision-making process.

---

## Monte Carlo Tree Search

The project architecture may use the idea of exploring multiple possible future actions and selecting the action with the best estimated score.

Monte Carlo Tree Search can be used conceptually to:

1. Start from the learner's current state.
2. Explore possible learning actions.
3. Simulate possible future learner states.
4. Evaluate the simulated outcomes.
5. Select the action with the highest estimated score.

However, Monte Carlo Tree Search should only be described as **implemented** if the actual project code contains a working MCTS algorithm.

If the current implementation only compares possible actions using the transition model, it should be described as:

> Future-action comparison and simulation using the action transition model.

It should not be claimed as a complete MCTS implementation unless the required tree-search, selection, expansion, simulation, and backpropagation steps are present in the code.

---

## Project Structure

The project is organized into the following main components:

```text
ACDT_Project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│
├── notebooks/
│
├── preprocessing/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

The exact files and folders may vary depending on the current implementation.

---

## Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* TensorFlow/Keras
* Logistic Regression
* Random Forest
* LSTM

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Development Tools

* Jupyter Notebook
* Google Colab
* Visual Studio Code
* Git
* GitHub

---

## Evaluation

The models can be evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

The project also includes visual analysis such as:

* Model comparison graphs
* Confusion matrices
* Learner-state distributions
* State-transition graphs
* Action recommendation graphs
* Learning trajectory visualizations

Because learner-state datasets may be imbalanced, accuracy should not be considered the only evaluation metric. Precision, recall, and F1-score are also important.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/fowzia-nilofer/ACDT_Project.git
```

Move into the project directory:

```bash
cd ACDT_Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Download the ASSISTments 2009–2010 Skill Builder Dataset.
2. Place the raw CSV file inside:

```text
data/raw/
```

3. Run the preprocessing scripts or notebooks.
4. Generate the processed learner features.
5. Train the machine learning models.
6. Generate learner-state predictions.
7. Run the action transition and future simulation modules.
8. Generate personalized learning recommendations.
9. Review the evaluation reports and visualizations.

The exact execution order may depend on the scripts included in the repository.

---

## Current Project Scope

The current project focuses on:

* Learner behavior analysis
* Performance-based learner-state estimation
* Sequential learner prediction
* Personalized learning recommendations
* Action transition modeling
* Future-state simulation

The current implementation does not directly measure emotions using sensors, facial expressions, physiological signals, or self-reported affect data.

Therefore, the affective component is currently represented through **behavioral proxies** derived from learner interactions.

---

## Future Enhancements

Possible future improvements include:

* Integrating the ASSISTments 2012–13 School Data with Affect
* Adding direct affect-related features such as boredom, frustration, confusion, and engagement
* Implementing a complete Monte Carlo Tree Search algorithm
* Comparing Monte Carlo Tree Search with Reinforcement Learning
* Adding a real-time learner dashboard
* Integrating an intelligent tutoring interface
* Using explainable AI for recommendations
* Improving transition-probability estimation
* Testing the system with additional educational datasets
* Building a more complete digital-twin simulation environment

---

## Conclusion

This project presents a predictive personalized learning framework that combines machine learning, learner-state estimation, action transition modeling, and future-state simulation.

By analyzing learner behavior and predicting possible future outcomes, the system aims to recommend suitable learning actions before learning difficulties become more serious.

The current implementation uses behavioral and performance-based proxies for affective-cognitive state estimation. Future work can strengthen the affective component by integrating datasets that contain direct affect-related measurements.

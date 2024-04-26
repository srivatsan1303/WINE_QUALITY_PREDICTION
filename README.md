# Wine Quality Prediction Using Apache Spark
## CS 643-Cloud Computing 

## Overview
This README provides instructions for setting up and executing a wine quality prediction model using Spark across four EC2 instances. The project utilizes AWS technologies, with a focus on parallel machine learning tasks using Apache Spark, EC2, S3, and Docker.

## Getting Started with AWS Academy
1. **Registration:** Access the AWS Academy course and sign up using your NJIT email.
2. **Account Setup:** Create an AWS account following the instructions provided via invitation links if you do not already have one.

## Environment Setup on AWS
### Initial Setup
1. Sign into AWS Academy and navigate to "Modules".
2. Initiate the "AWS Academy Learner Lab" and ensure it is active (indicated by a green circle).
3. From the "AWS Details" section, document your AWS access, secret keys, and session token. Secure the PEM file from "SSH key" for EC2 access.

### Configuring AWS Resources
#### EMR and EC2 Setup
1. **Cluster Creation:**
   - Initiate by clicking "Create cluster".
   - **Software Setup:** Opt for necessary applications like Hadoop and Spark.
   - **Hardware Setup:** Designate instances for master and core nodes, and assign four task nodes.
2. **Cluster Configuration:**
   - Name your cluster and adjust scaling provisions.
   - Assign "key1" as the EC2 key pair for SSH purposes.
   - Optionally enable settings for enhanced logging and debugging.
3. **Roles Configuration:** Apply `EMR_EC2_DefaultRole` and `EMR_AutoScaling_DefaultRole` to appropriate roles.

## Execution of Machine Learning Model

The provided Python script executes the following key steps using Apache Spark to predict wine quality:

1. **Spark Session Initialization:** A Spark session is created to facilitate all subsequent operations.
2. **Data Loading:** The script loads training and validation datasets from CSV files, handling schema inference and setting headers appropriately.
3. **Data Preprocessing:** Columns in both datasets are normalized to double type and unnecessary quotes are removed.
4. **Feature Engineering and Model Setup:** Features for predicting wine quality are assembled and indexed, followed by the configuration of a RandomForest classifier.
5. **Model Training and Evaluation:** The model is trained on the preprocessed training data, then predictions are made on the validation data. These predictions are evaluated using accuracy and F1 score metrics.
6. **Hyperparameter Tuning:** Cross-validation is used to fine-tune the model parameters, aiming to optimize the prediction performance.
7. **Final Model Evaluation:** The best model from cross-validation is evaluated on the validation dataset to determine the final accuracy and F1 scores.

The script is designed to be robust and efficient, leveraging caching and a comprehensive pipeline to ensure high performance and reliable results.

## Results and Metrics Without Using Docker
- **Accuracy:** 0.96875
- **Weighted F1 Score:** 0.954190

## Docker Configuration and Usage
- **Build Docker Image:** `docker build -t srivatsan1303/programming_assignment_2_wine_prediction:wineprediction .`
- **Push to Docker Hub:** `docker push srivatsan1303/programming_assignment_2_wine_prediction:wineprediction`
- **Manage Docker Services:** Start, enable, and verify Docker using systemctl.
- **Run Docker Container:** `sudo docker run srivatsan1303/programming_assignment_2_wine_prediction:wineprediction

### Additional Docker Information
**Docker Image Details:** [Visit_Docker](https://hub.docker.com/layers/srivatsan1303/programming_assignment_2_wine_prediction/wineprediction/images/sha256-449699db4e3d40eac2c0a29f7bf662ef7f1123ee54ea60a323a41d47f0a00af2?context=repo).

## Project Repository
**GitHub:** [Visit Repository](https://github.com/srivatsan1303/WINE_QUALITY_PREDICTION)

### Final Model Performance
- **Model Accuracy:** 0.967
- **F1 Score:** 0.954

- ### Student Details
**Name:** Srivatsan Jayaraman
**UCID:** sj796 
**Contact:** sj796@njit.edu 

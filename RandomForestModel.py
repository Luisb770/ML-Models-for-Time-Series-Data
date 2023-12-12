import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Step 2: Load and Prepare Data
data_1 = pd.read_csv(r'C:\Users\bravo\OneDrive\OneDrive Files\Desktop\train_set_1.csv')
data_2 = pd.read_csv(r'C:\Users\bravo\OneDrive\OneDrive Files\Desktop\train_set_2.csv')
data_3 = pd.read_csv(r'C:\Users\bravo\OneDrive\OneDrive Files\Desktop\train_set_3.csv')

# Step 3: Generate Features for Financial Time Series Data
def generate_features(data):
    lag = 5
    data['SMA_5'] = data['value'].rolling(window=5).mean()
    data['SMA_20'] = data['value'].rolling(window=20).mean()

    for i in range(1, lag + 1):
        data[f'Lag_{i}'] = data['value'].shift(i)
    
    data['Rolling_STD_5'] = data['value'].rolling(window=5).std()
    data['Rolling_STD_20'] = data['value'].rolling(window=20).std()
    
    roc_period = 1
    data['ROC'] = (data['value'].diff(roc_period).shift(-1) > 0).astype(int)  # Shift ROC as required

    return data

data_1 = generate_features(data_1)
data_2 = generate_features(data_2)
data_3 = generate_features(data_3)

# Step 4: Prepare Features and Labels for all Datasets
def prepare_data(data):
    lag = 5
    data = data.dropna()
    
    X = data[['SMA_5', 'SMA_20', 'Rolling_STD_5', 'Rolling_STD_20'] + [f'Lag_{i}' for i in range(1, lag + 1)]]
    y = data['ROC']

    return X, y

X_1, y_1 = prepare_data(data_1)
X_2, y_2 = prepare_data(data_2)
X_3, y_3 = prepare_data(data_3)

# Step 5: Split Data into Training and Test Sets for all Datasets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

# Modify the create_random_forest_model function to create a random forest model
def create_random_forest_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# Modify the train_and_evaluate_random_forest_model function to train and evaluate the random forest model
def train_and_evaluate_random_forest_model(X_train, y_train, X_test, y_test):
    model = create_random_forest_model()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    return model

# Step 5: Split Data into Training and Test Sets for all Datasets
X_train_1, X_test_1, y_train_1, y_test_1 = split_data(X_1, y_1)

# Now, when you call the evaluation function, it will print MSE, RMSE, and create a training history plot.
print("Evaluation for Dataset 1:")
model_1 = train_and_evaluate_random_forest_model(X_train_1, y_train_1, X_test_1, y_test_1)

# Step 5: Split Data into Training and Test Sets for all Datasets
X_train_2, X_test_2, y_train_2, y_test_2 = split_data(X_2, y_2)

# Now, when you call the evaluation function, it will print MSE, RMSE, and create a training history plot.
print("Evaluation for Dataset 2:")
model_2 = train_and_evaluate_random_forest_model(X_train_2, y_train_2, X_test_2, y_test_2)

# Step 5: Split Data into Training and Test Sets for all Datasets
X_train_3, X_test_3, y_train_3, y_test_3 = split_data(X_3, y_3)

# Now, when you call the evaluation function, it will print MSE, RMSE, and create a training history plot.
print("Evaluation for Dataset 3:")
model_3 = train_and_evaluate_random_forest_model(X_train_3, y_train_3, X_test_3, y_test_3)

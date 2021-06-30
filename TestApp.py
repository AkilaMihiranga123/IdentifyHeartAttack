# Import required libraries
import numpy as np
import joblib

# Sample test data
test_data = [62, 0, 160, 0, 145, 98.2]

# convert test_data into numpy array
test_data_array = np.array(test_data)

# Reshape the data
reshaped_test_data = test_data_array.reshape(1,-1)
print(reshaped_test_data)

# Load trained model
trained_model = joblib.load('./IdentifyHeartAttackModel/identify_heart_attack_randomforest_model.joblib')

# Predict using sample test data
prediction = trained_model.predict(reshaped_test_data)
print(prediction)
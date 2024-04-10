import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
num_students = 10000

# Generate student data
data = pd.DataFrame({
    'Roll No': range(1, num_students + 1),
    'Gender': np.random.choice(['Male', 'Female'], size=num_students),
    '10th Percentage': np.random.randint(0, 101, size=num_students),
    '12th Percentage': np.random.randint(0, 101, size=num_students),
    '12th Stream': np.random.choice(['Science', 'Art', 'Commerce'], size=num_students),
    'Family Count': np.random.randint(0, 11, size=num_students),
    'Father Occupation': np.random.choice(['Government Job', 'Private Job', 'Army', 'Housewife', 'Business', 'Farmer', 'No Job'], size=num_students),
    'Mother Occupation': np.random.choice(['Government Job', 'Private Job', 'Army', 'Housewife', 'Business', 'Farmer', 'No Job'], size=num_students),
    'Siblings Count': np.random.randint(0, 6, size=num_students),
})

# Generate random subject marks for each semester
subjects_all_sem = [
    ['English', 'Math', 'Basic Science', 'ICT', 'WPC'],
    ['EEC', 'AMI', 'BEC', 'PCI', 'BCC', 'CPH', 'WPD'],
    ['OOP', 'DSU', 'CGR', 'DMS', 'DTE'],
    ['JPR', 'SEN', 'DCC', 'MIC', 'GAD']
]

for i, subjects in enumerate(subjects_all_sem):
    data = pd.concat([data, pd.DataFrame(np.random.randint(0, 101, size=(num_students, len(subjects))), columns=[f"{subject}_{i+1}st_Sem" for subject in subjects])], axis=1)

# Generate 5th semester overall percentage (target variable)
data['Overall Percentage (5th Sem)'] = np.random.randint(0, 101, size=num_students)

# Assign target groups using numpy.select
conditions = [
    data['Overall Percentage (5th Sem)'] < 35,
    (data['Overall Percentage (5th Sem)'] >= 35) & (data['Overall Percentage (5th Sem)'] < 50),
    (data['Overall Percentage (5th Sem)'] >= 50) & (data['Overall Percentage (5th Sem)'] < 75),
    data['Overall Percentage (5th Sem)'] >= 75
]
choices = ['<35', '35 to 50', '50 to 75', '75 to 100']
data['Target Group'] = np.select(conditions, choices, default='NA')  # Set a default value if none of the conditions are met

# Map marks to target groups
def map_marks_to_groups(row):
    marks_cols = [col for col in data.columns if col.endswith('Sem')]
    for col in marks_cols:
        row[col] = np.minimum(row[col] * 0.1 + 5 if row['Target Group'] == '<35' else
                               row[col] * 0.4 + 50 if row['Target Group'] == '35 to 50' else
                               row[col] * 0.7 + 75 if row['Target Group'] == '50 to 75' else
                               row[col], 100)  # Ensure marks don't exceed 100
    return row

data = data.apply(map_marks_to_groups, axis=1)

# Save the dataframe to a CSV file
data.to_csv('dummy_student_data_10000.csv', index=False)
print("Dummy student data generated and saved to 'dummy_student_data_10000.csv'.")


gender_mapping = {'Male': 1, 'Female': 0}
data['Gender'] = data['Gender'].map(gender_mapping)

stream_12_mapping = {'Science': 2, 'Commerce': 1,'Art':0}
data['12th Stream'] = data['12th Stream'].map(stream_12_mapping)

modelling_df=data.copy()

modelling_df.drop(columns=['Roll No','Mother Occupation','Overall Percentage (5th Sem)'],axis=1,inplace=True)
modelling_df.shape

target_mapping = {'35 to 50':1, '<35':0, '75 to 100':3, '50 to 75':2}
modelling_df['Target Group'] = modelling_df['Target Group'].map(target_mapping)

modelling_df = pd.get_dummies(modelling_df)
modelling_df.shape

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define features (X) and target variable (y)
X = modelling_df.drop(['Target Group'], axis=1)  # Features
y = modelling_df['Target Group']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = XGBClassifier()
# Train the model
model.fit(X_train, y_train)
# Make predictions on the testing data
predictions = model.predict(X_test)


# Classification report
print("Classification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

modelling_df.head()


import joblib
# Assuming your XGBoost model is named 'model'
# Save the model to a pickle file
joblib.dump(model, 'xgboost_model.pkl')
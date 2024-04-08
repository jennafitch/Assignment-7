import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Read the data from the Excel file
data = pd.read_excel(r'C:\Users\jenfi\Downloads\baseball.xlsx')

# Step 2: Preprocess the data
# Extract dependent variables (features) and independent variable (target)
X = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
y = data['Playoffs']

# Step 3: Calculate correlation coefficients for each dependent variable
correlation_coefficients = X.corrwith(y)

# Print correlation coefficients
print("\nCorrelation Coefficients:")
print(correlation_coefficients)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)

# Step 7: Compare predicted results to actual data for the year 2012
# Filter data for the year 2012
data_2012 = data[data['Year'] == 2012]
X_2012 = data_2012[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
y_actual_2012 = data_2012['Playoffs']

# Predict playoff status for the year 2012
y_pred_2012 = model.predict(X_2012)

# Step 8: Print predicted and actual results for each team in 2012
print("\nPredicted vs Actual Results for the year 2012:")
for index, row in data_2012.iterrows():
    team_name = row['Team']
    predicted_playoffs = "Made Playoffs" if y_pred_2012[index - data_2012.index[0]] == 1 else "Did not make Playoffs"
    actual_playoffs = "Made Playoffs" if y_actual_2012[index] == 1 else "Did not make Playoffs"
    print(f"{team_name}: Predicted - {predicted_playoffs}, Actual - {actual_playoffs}")
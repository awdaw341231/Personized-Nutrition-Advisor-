import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template

# Expanded dataset: Nutritional needs based on user profiles
data = {
    'Age': [25, 30, 35, 40, 22, 29, 33, 50, 18, 45],
    'Weight': [70, 80, 90, 100, 60, 85, 95, 110, 55, 105],
    'Height': [175, 180, 165, 170, 160, 178, 182, 172, 158, 168],
    'ActivityLevel': [1, 2, 3, 1, 2, 3, 2, 1, 3, 2],  # 1: Low, 2: Medium, 3: High
    'HealthGoal': ['Lose Weight', 'Gain Muscle', 'Maintain Weight', 'Lose Weight', 'Gain Muscle', 'Lose Weight', 'Gain Muscle', 'Maintain Weight', 'Lose Weight', 'Gain Muscle'],
    'DietPlan': ['Low Carb', 'High Protein', 'Balanced', 'Low Fat', 'High Protein', 'Low Carb', 'High Protein', 'Balanced', 'Low Fat', 'High Protein']
}

df = pd.DataFrame(data)

# Encoding categorical data
health_goal_mapping = {goal: i for i, goal in enumerate(df['HealthGoal'].unique())}
df['HealthGoal'] = df['HealthGoal'].map(health_goal_mapping)
diet_plan_mapping = {i: label for i, label in enumerate(df['DietPlan'].unique())}
df['DietPlan'] = df['DietPlan'].astype('category').cat.codes

# Splitting dataset
X = df[['Age', 'Weight', 'Height', 'ActivityLevel', 'HealthGoal']]
y = df['DietPlan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Flask app setup
app = Flask(__name__)

# Function for personalized recommendation
def recommend_nutrition(age, weight, height, activity_level, health_goal):
    if health_goal not in health_goal_mapping:
        raise ValueError(f"Invalid health goal: {health_goal}. Supported goals are {list(health_goal_mapping.keys())}")
    health_goal_encoded = health_goal_mapping[health_goal]
    user_input = np.array([[age, weight, height, activity_level, health_goal_encoded]])
    diet_code = model.predict(user_input)[0]
    return diet_plan_mapping[diet_code]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    age = int(data.get('age'))
    weight = int(data.get('weight'))
    height = int(data.get('height'))
    activity_level = int(data.get('activity_level'))
    health_goal = data.get('health_goal')
    try:
        recommendation = recommend_nutrition(age, weight, height, activity_level, health_goal)
        return jsonify({'recommendation': recommendation})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

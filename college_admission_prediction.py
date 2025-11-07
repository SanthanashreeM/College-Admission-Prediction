import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("admission_data.csv")

# Encode categorical columns
le_category = LabelEncoder()
le_branch = LabelEncoder()

data["category"] = le_category.fit_transform(data["category"])
data["branch"] = le_branch.fit_transform(data["branch"])

# Prepare input/output
X = data.drop("admitted", axis=1)
y = data["admitted"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, pred))

# ----------------------------
# USER INPUT SECTION
# ----------------------------

print("\n--- College Admission Prediction ---")
tenth = float(input("Enter 10th percentage: "))
twelfth = float(input("Enter 12th percentage: "))
entrance = float(input("Enter entrance exam score: "))
category = input("Enter category (General/OBC/SC/ST): ")
branch = input("Enter branch: ")

# Encode user inputs
category_encoded = le_category.transform([category])[0]
branch_encoded = le_branch.transform([branch])[0]

# ✅ Prepare input as a DataFrame to remove warning
user_data = pd.DataFrame([{
    "tenth": tenth,
    "twelfth": twelfth,
    "entrance_score": entrance,
    "category": category_encoded,
    "branch": branch_encoded
}])

# Prediction probability
prob = model.predict_proba(user_data)[0][1]

print("\nPrediction Probability:", round(prob, 2))

# Interpretation
if prob >= 0.75:
    print("High Chance of Admission!")
elif prob >= 0.40:
    print("Moderate Chance of Admission.")
else:
    print("Low Chance of Admission.")

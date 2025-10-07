import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump

# Load the dataset
df = pd.read_csv('C:/Users/karan/OneDrive/Desktop/Final IDS/ddos_traffic.csv')  # Replace with your dataset path

# Data exploration
print("Dataset shape:", df.shape)

# Strip any extra spaces in column names (if any)
df.columns = df.columns.str.strip()

# Drop rows where any value is infinite
df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

# Handle missing values
df.dropna(inplace=True)  # Drop rows with NaN values

# Top features including new columns
top_features = ['Avg Bwd Segment Size', 'Total Length of Bwd Packets', 
                'Bwd Packet Length Max', 'Flow IAT Mean', 'Bwd Packet Length Mean',
                'Flow Packets/s', 'Bwd Packets/s', 'Fwd IAT Total', 'Flow IAT Max', 'Destination Port']

# Feature selection (splitting dataset into features and target)
#X = df.drop('Label', axis=1)  # Drop the 'Label' column
X = df[top_features]
y = df['Label']  # Target variable

print(y.value_counts())
print(X.columns)

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Verify class distribution before and after undersampling
print("Class distribution before undersampling:", dict(pd.Series(y).value_counts()))
print("Class distribution after undersampling:", dict(pd.Series(y_resampled).value_counts()))

# Scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Function to calculate and display metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=1):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=1):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Model 1: Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rfc, X_train, X_test, y_train, y_test)

# Model 2: Decision Tree
dtc = DecisionTreeClassifier(random_state=42)
evaluate_model(dtc, X_train, X_test, y_train, y_test)

# Model 3: Support Vector Machine
svm_model = svm.SVC(kernel='linear', random_state=42)
evaluate_model(svm_model, X_train, X_test, y_train, y_test)

# Model 4: k-Nearest Neighbors
knn = KNeighborsClassifier()
evaluate_model(knn, X_train, X_test, y_train, y_test)

# Save models
dump(rfc, 'random_forest_model1.joblib')  # Model
dump(scaler, 'scaler1.joblib')            # StandardScaler
dump(label_encoder, 'label_encoder1.joblib')  # LabelEncoder

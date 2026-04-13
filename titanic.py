import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = sns.load_dataset('titanic')
print("Dataset loaded! Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


df['age'].fillna(df['age'].median(), inplace=True)


df.drop(columns=['deck', 'embark_town', 'alive', 'who',
                 'adult_male', 'class'], inplace=True)


df.dropna(inplace=True)

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nAfter cleaning, shape:", df.shape)


X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did not survive', 'Survived'],
            yticklabels=['Did not survive', 'Survived'])
plt.title('Titanic Survival Prediction - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results.png')
print("Chart saved as results.png")


plt.figure(figsize=(8, 5))
feat_importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

sns.barplot(x=feat_importance.values, y=feat_importance.index, palette='viridis')
plt.title('What factors determined survival?')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance chart saved!")

print("\nDone! Check your folder for results.png and feature_importance.png")
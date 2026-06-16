import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
csv_file = "24_Coffee_Taste_Test.csv"
print("☕ Loading coffee data for Machine Learning...")

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"❌ Error: '{csv_file}' not found.")
    exit()

# 2. Data Preprocessing (データの前処理)
# 予測したい答え（目的変数）は「一番好きなコーヒー（prefer_overall）」
# 答えが空欄（NaN）の人は学習に使えないので除外
df_ml = df.dropna(subset=['prefer_overall']).copy()

# 予測の手掛かり（説明変数）として使う列をピックアップ
# 年齢、飲む杯数、在宅勤務、飲む場所、抽出方法(brew) を採用
features = ['age', 'cups', 'wfh', 'where_drink', 'brew']
X_raw = df_ml[features].fillna('Unknown') # 空欄は'Unknown'として埋める
y = df_ml['prefer_overall']

# ★AIは文字を読めないので、文字データを「0と1の数値」に変換する（One-Hot Encoding）
X = pd.get_dummies(X_raw, drop_first=True)

# 3. Train-Test Split (学習用とテスト用に分割)
# 全データの80%を学習（教科書）に、20%をテスト（実力テスト）に分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📚 Training Data: {len(X_train)} people")
print(f"📝 Testing Data: {len(X_test)} people")

# 4. Train the Model (AIの学習スタート！)
print("🧠 Training Random Forest AI... (This will be very fast!)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train) # ここで一瞬で学習が完了します

# 5. Make Predictions & Evaluate (テストと採点)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"🎯 AI Prediction Accuracy: {accuracy * 100:.2f}%")
print("-" * 40)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Feature Importance (AIは何を重要視して予測したか？)
# AIが好みを予測する上で、どのデータが一番「役に立ったか」を可視化する
importances = rf_model.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10) # 上位10個

# グラフ描画
sns.set_theme(style="whitegrid", palette="muted")
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_df, x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Most Important Factors in Choosing a Coffee", fontsize=14, fontweight='bold')
plt.xlabel("Impact on AI's Decision")
plt.ylabel("Factors (Features)")
plt.tight_layout()
plt.show()
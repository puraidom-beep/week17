import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
csv_file = "24_Coffee_Taste_Test.csv"
print("Loading coffee data...")

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Successfully loaded! Total {len(df)} respondents.")
except FileNotFoundError:
    print(f"❌ Error: '{csv_file}' not found. Please check the file path.")
    exit()

# 2. Set the aesthetic style of the plots
sns.set_theme(style="whitegrid", palette="muted")

# サイズ設定
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
fig.suptitle("The Great American Coffee Taste Test 2024 - Data Exploration", fontsize=16, fontweight='bold')

# --- Chart 1: Age Distribution ---
sns.countplot(data=df, y='age', ax=axes[0, 0], palette="Blues_d",
              order=df['age'].value_counts().index)
axes[0, 0].set_title("1. Demographics: Age Distribution", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Number of Respondents")
axes[0, 0].set_ylabel("Age Group")

# --- Chart 2: Cups per Day ---
sns.countplot(data=df, x='cups', ax=axes[0, 1], palette="flare",
              order=['Less than 1', '1', '2', '3', '4', 'More than 4'])
axes[0, 1].set_title("2. Coffee Habits: Cups per Day", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Cups")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_xticklabels(['<1', '1', '2', '3', '4', '4<'])

# --- Chart 3: Blind Taste Test Preference ---
if 'prefer_overall' in df.columns:
    sns.countplot(data=df, x='prefer_overall', ax=axes[1, 0], palette="Set2",
                  order=df['prefer_overall'].value_counts().index)
    axes[1, 0].set_title("3. Blind Test: Overall Favorite Coffee", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Coffee Choice (A/B/C/D)")
    axes[1, 0].set_ylabel("Votes")
else:
    axes[1, 0].text(0.5, 0.5, 'Column "prefer_overall" not found', ha='center')

# --- Chart 4: Work From Home vs Drinking at Home ---
if 'wfh' in df.columns and 'where_drink' in df.columns:
    df['drink_at_home'] = df['where_drink'].astype(str).apply(lambda x: 'Yes' if 'At home' in x else 'No')

    wfh_map = {
        'I primarily work from home': 'WFH',
        'I do a mix of both': 'Hybrid',
        'I primarily work in person': 'In Person'
    }
    df['wfh_short'] = df['wfh'].map(lambda x: wfh_map.get(x, str(x)))

    sns.countplot(data=df, x='wfh_short', hue='drink_at_home', ax=axes[1, 1], palette="pastel",
                  order=['WFH', 'Hybrid', 'In Person'])

    axes[1, 1].set_title("4. Work From Home vs Drinking at Home", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Work Style")
    axes[1, 1].set_ylabel("Count")
else:
    axes[1, 1].text(0.5, 0.5, 'Columns for WFH not found', ha='center')

plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
fig.subplots_adjust(top=0.88)

# Display the dashboard
plt.show()
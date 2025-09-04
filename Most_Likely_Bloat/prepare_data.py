import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print("Loading and creating a BALANCED 80/20 data split...")
df = pd.read_csv("public_train.csv")
df.dropna(subset=['post_message'], inplace=True)

# Separate by label
df_true = df[df['label'] == 0]
df_fake = df[df['label'] == 1]

# Create initial splits
train_true, test_true = train_test_split(df_true, test_size=0.2, random_state=42)
train_fake, test_fake = train_test_split(df_fake, test_size=0.2, random_state=42)

# Create and save the imbalanced test set (to reflect reality)
test_df = pd.concat([test_true, test_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df.to_csv("test_data.csv", index=False)
print(f"Test data saved to test_data.csv ({len(test_df)} rows)")

# Balance the TRAINING data using oversampling
minority_oversampled = resample(train_fake,
                                replace=True,
                                n_samples=len(train_true),
                                random_state=42)

balanced_train_df = pd.concat([train_true, minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced_train_df.to_csv("train_data.csv", index=False)
print(f"BALANCED training data saved to train_data.csv ({len(balanced_train_df)} rows)")
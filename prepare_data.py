import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
print("Loading public_train.csv...")
df = pd.read_csv("public_train.csv")

# Drop rows with missing post messages
df.dropna(subset=['post_message'], inplace=True)

# Separate the data by label
df_true = df[df['label'] == 0]
df_fake = df[df['label'] == 1]

# Split the true news (80% train, 20% test)
train_true, test_true = train_test_split(df_true, test_size=0.2, random_state=42)

# Split the fake news (80% train, 20% test)
train_fake, test_fake = train_test_split(df_fake, test_size=0.2, random_state=42)

# Combine the splits to create the final training and testing sets
train_df = pd.concat([train_true, train_fake])
test_df = pd.concat([test_true, test_fake])

# Shuffle the datasets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Data splitting complete!")
print(f"Training data saved to train_data.csv ({len(train_df)} rows)")
print(f"Testing data saved to test_data.csv ({len(test_df)} rows)")
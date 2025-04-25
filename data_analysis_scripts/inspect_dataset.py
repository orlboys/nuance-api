import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("training\data\political_bias.csv")

def label_distribution(df):
    """
    Analyzes and visualizes the distribution of labels in the dataset.
    Important because:
    - Reveals class imbalance issues that could bias the model
    - Helps determine if we need data augmentation or resampling
    - Indicates the natural distribution of political bias in the dataset
    """
    label_counts = df['label'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def text_length_distribution(df):
    """
    Analyzes the distribution of text lengths in the dataset.
    Important because:
    - Helps identify potential preprocessing needs (truncation/padding)
    - Reveals outliers in text length that might affect model performance
    - Guides decisions about model architecture and input handling
    """
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    print(df['word_count'].describe())
    plt.figure(figsize=(8, 6))
    sns.histplot(df['word_count'], bins=30, kde=True, color='skyblue')
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()

def class_proportions(df):
    """
    Calculates the proportion of each class in the dataset.
    Important because:
    - Helps determine appropriate evaluation metrics
    - Guides decisions about class weights in model training
    - Indicates if stratification is needed in train/test splits
    """
    label_proportions = df['label'].value_counts(normalize=True).sort_index()
    print(label_proportions)

def display_sample_texts(df):
    """
    Displays random samples of texts from each label category.
    Important because:
    - Provides qualitative insights into the data
    - Helps identify potential data quality issues
    - Reveals patterns in how political bias manifests in text
    """
    for label in sorted(df['label'].unique()):
        print(f"\nLabel {label} Samples:")
        print(df[df['label'] == label]['text'].sample(3, random_state=42).to_string(index=False))

def check_missing_values(df):
    """
    Checks for missing values and unexpected labels in the dataset.
    Important because:
    - Missing values can impact model training and performance
    - Unexpected labels could indicate data quality issues
    - Ensures data integrity before model development
    """
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])

    expected_labels = set(range(5))
    actual_labels = set(df['label'].unique())
    unexpected_labels = actual_labels - expected_labels
    print("\nUnexpected Labels:")
    print(unexpected_labels)

print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe(include='all'))
print("Dataset Shape:")
print(df.shape)
print("\nLabel Counts:")
print(df['label'].value_counts())
print("\nLabel Distribution:")
label_distribution(df)
print("\nText Length Distribution:")
text_length_distribution(df)
print("\nClass Proportions:")
class_proportions(df)
print("\nSample Texts:")
display_sample_texts(df)
print("\nMissing Values and Unexpected Labels:")
check_missing_values(df)
# The above code provides a comprehensive analysis of the dataset, including label distribution, text length distribution, class proportions, sample texts, and checks for missing values and unexpected labels. This information is crucial for understanding the dataset and preparing it for model training.
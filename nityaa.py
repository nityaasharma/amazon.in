import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (assuming it's a CSV file)
file_path ='C:\Users\PC\Downloads\amazon dataset'  
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())

df = df.dropna(subset=['actual_price', 'rating'], axis=0) 


df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Fill missing values in categorical columns (like product category or brand) with a placeholder
df['category'].fillna('Unknown', inplace=True)
df['product_name'].fillna('Unknown', inplace=True)

# Data Analysis:

## 1. Product Count by Category
category_count = df['category'].value_counts()

# Plotting product count by category
plt.figure(figsize=(10, 6))
sns.countplot(y='category', data=df, order=category_count.index, palette='viridis')
plt.title('Product Count by Category')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()

## 2. Distribution of Product Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['actual_price'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Product Prices')
plt.xlabel('actual_price')
plt.ylabel('Frequency')
plt.show()

## 3. Average Rating by Product Category
avg_rating_by_category = df.groupby('category')['rating'].mean().sort_values(ascending=False)

# Plotting average rating by category
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_rating_by_category.values, y=avg_rating_by_category.index, palette='coolwarm')
plt.title('Average Rating by Product Category')
plt.xlabel('Average Rating')
plt.ylabel('Category')
plt.show()

## 4. Top 10 Most Expensive Products
top_10_expensive = df.sort_values(by='actual_price', ascending=False).head(10)

print("\nTop 10 Most Expensive Products:")
print(top_10_expensive[['product_name', 'price']])

# Plotting Top 10 Expensive Products
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_expensive['actual_price'], y=top_10_expensive['product_name'], palette='magma')
plt.title('Top 10 Most Expensive Products')
plt.xlabel('actual_price')
plt.ylabel('product_name')
plt.show()

## 5. Correlation Heatmap of Numeric Features
# Check correlations between numeric features (like price, rating, etc.)
corr_matrix = df[['actual_price', 'rating']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap between Price and Rating')
plt.show()

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_amazon_in_data.csv', index=False)

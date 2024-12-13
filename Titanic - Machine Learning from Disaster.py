# Project Titanic - Machine Learning from Disaster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Đặt PassengerId làm chỉ số
train_df.set_index('PassengerId', inplace=True)

# Hiển thị thông tin mô tả
print("Mô tả dữ liệu train:")
print(train_df.describe())

# Hiển thị thông tin tổng quát
print("\nThông tin dữ liệu train:")
print(train_df.info())

# Kiểm tra số giá trị bị thiếu trong cột Age
print("\nSố giá trị bị thiếu trong cột Age:")
print(train_df['Age'].isnull().sum())

# Điền giá trị thiếu trong cột Age (nếu cần)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Phân nhóm tuổi
bins = [0, 12, 18, 30, 50, 80]
labels = ['Trẻ em', 'Thiếu niên', 'Thanh niên', 'Trung niên', 'Người già']
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=bins, labels=labels)

# Biểu đồ trực quan hóa số lượng hành khách theo nhóm tuổi
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Age_Group', palette='viridis')
plt.title("Số lượng hành khách theo nhóm tuổi")
plt.xlabel("Nhóm tuổi")
plt.ylabel("Số lượng")
plt.show()

# Hiển thị tên cột
print("\nTên các cột trong train_df:")
print(train_df.columns)

print("\nTên các cột trong test_df:")
print(test_df.columns)

# Hiển thị 5 dòng đầu tiên
print("\n5 dòng đầu tiên trong train_df:")
print(train_df.head())

print("\n5 dòng đầu tiên trong test_df:")
print(test_df.head())

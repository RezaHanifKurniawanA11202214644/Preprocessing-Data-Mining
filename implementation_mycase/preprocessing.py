# Importing the libraries
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset dan membagi data ke variabel X sebagai attribute reguler dan y sebagai attribute label
dataset = pd.read_csv('gym_members.csv')
# Membagi data ke variabel X sebagai attribute reguler dan y sebagai attribute label
X = dataset.drop(dataset.columns[9], axis=1).values
y = dataset.iloc[:, 9].values
# Menampilkan data yang sudah dibagi
print("Data variabel X sebagai attribute reguler:\n", X)
print("\nData variabel Y sebagai attribute label:\n", y)
print("\n===================================================================================================\n")


# Mengatasi missing value pada variabel X (attribute reguler) menggunakan SimpleImputer
from sklearn.impute import SimpleImputer
for i in range(14):
    if pd.api.types.is_numeric_dtype(X):
        imputer_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_numeric.fit(X[:, i:i+1])
        X[:, i:i+1] = imputer_numeric.transform(X[:, i:i+1])
    else:
        imputer_non_numeric = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer_non_numeric.fit(X[:, i:i+1])
        X[:, i:i+1] = imputer_non_numeric.transform(X[:, i:i+1])
# Menampilkan data yang sudah diatasi
print("\nData variabel X setelah diatasi missing valuenya:\n", X)
# Mengatasi missing value pada variabel y (attribute label) menggunakan SimpleImputer
imputer_label = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_label.fit(y.reshape(-1, 1))
y = imputer_label.transform(y.reshape(-1, 1)).ravel()
# Menampilkan data yang sudah diatasi
print("\nData variabel Y setelah diatasi missing valuenya:\n", y)
print("\n===================================================================================================\n")


# Encoding data kategori ke numerik pada variabel X (attribute reguler) menggunakan OneHotEncoder dan ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Menampilkan data yang sudah di encoding
print("\nData variabel X setelah di encoding:\n", X)


# Encoding data kategori ke numerik pada variabel y (attribute label) menggunakan OneHotEncoder dan ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Menampilkan data yang sudah di encoding
print("\nData variabel Y setelah di encoding:\n", y)
print("\n===================================================================================================\n")


# Membagi data menjadi data training dan data testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Menampilkan data training dan data testing
print("\nData training:\n", X_train)
print("\nData testing:\n", X_test)
print("\nLabel training:\n", y_train)
print("\nLabel testing:\n", y_test)
print("\n===================================================================================================\n")


# Feature Scaling atau normalisasi data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
X_test[:, 2:] = sc.transform(X_test[:, 2:])
# Menampilkan data training dan data testing
print("\nData training setelah di feature scaling:\n", X_train)
print("\nData testing setelah di feature scaling:\n", X_test)
print("\n")
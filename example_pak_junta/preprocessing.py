# Importing the libraries
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset dan membagi data ke variabel X sebagai attribute reguler dan y sebagai attribute label
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Menampilkan data yang sudah dibagi
print("Data variabel X sebagai attribute reguler:\n", X)
print("\nData variabel Y sebagai attribute label:\n", y)

# Mengatasi missing value dengan menggunakan SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# Menampilkan data yang sudah diatasi
print("\nData variabel X setelah diatasi:\n", X)

# Encoding data kategori ke numerik pada variabel X (attribute reguler)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Menampilkan data yang sudah di encoding
print("\nData variabel X setelah di encoding:\n", X)

# Encoding data kategori ke numerik pada variabel y (attribute label)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Menampilkan data yang sudah di encoding
print("\nData variabel Y setelah di encoding:\n", y)

# Membagi data menjadi data training dan data testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Menampilkan data training dan data testing
print("\nData training:\n", X_train)
print("\nData testing:\n", X_test)
print("\nLabel training:\n", y_train)
print("\nLabel testing:\n", y_test)

# Feature Scaling atau normalisasi data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
# Menampilkan data training dan data testing
print("\nData training setelah di feature scaling:\n", X_train)
print("\nData testing setelah di feature scaling:\n", X_test)
print("\n")
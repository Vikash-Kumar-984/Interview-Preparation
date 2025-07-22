# Interview-Preparation



# 📐 NumPy Cheatsheet: Numerical Operations

NumPy (Numerical Python) is the foundational library for numerical computing in Python. It provides high-performance multidimensional array objects (`ndarray`) and tools for working with these arrays.

**Import Convention:** `import numpy as np`

## 1. NumPy Arrays (`ndarray`)

* **Core Concept**: A **NumPy array** is a grid of values, all of the same type, and is indexed by a tuple of non-negative integers. It's significantly faster and more memory-efficient than Python lists for numerical operations.
    

## 2. Array Creation

* **From Python list/tuple**:
    ```python
    arr1d = np.array([1, 2, 3])
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    ```
* **Initial Placeholders**:
    ```python
    np.zeros((3, 4))      # 3x4 array of zeros
    np.ones((2, 3), dtype=int) # 2x3 array of ones (integers)
    np.empty((2, 2))      # 2x2 array, uninitialized (fast)
    np.full((2, 2), 7)    # 2x2 array filled with 7
    np.eye(3)             # 3x3 Identity matrix
    ```
* **Sequences**:
    ```python
    np.arange(0, 10, 2)   # Array from 0 to 9 with step 2: [0, 2, 4, 6, 8]
    np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1: [0.0, 0.25, 0.5, 0.75, 1.0]
    ```
* **Random Arrays**:
    ```python
    np.random.rand(2, 3)    # Uniformly distributed random numbers (0 to 1)
    np.random.randn(2, 3)   # Standard normal distribution (mean=0, std=1)
    np.random.randint(0, 10, size=(2, 2)) # Random integers (low=0, high=10)
    ```

---

## 3. Array Attributes

* `arr.shape`: Tuple of array dimensions (e.g., `(3, 4)`).
* `arr.ndim`: Number of array dimensions (rank).
* `arr.size`: Total number of elements in the array.
* `arr.dtype`: Data type of array elements (e.g., `int64`, `float32`).
* `arr.itemsize`: Size of each element in bytes.

---

## 4. Indexing & Slicing

* **Basic Indexing (1D)**:
    ```python
    arr = np.array([0, 1, 2, 3, 4])
    arr[0]      # 0
    arr[1:4]    # [1, 2, 3] (slice, end exclusive)
    arr[::-1]   # [4, 3, 2, 1, 0] (reverse)
    ```
* **2D Indexing (Row, Column)**:
    ```python
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr2d[0, 1]     # 2 (row 0, col 1)
    arr2d[0, :]     # [1, 2, 3] (entire row 0)
    arr2d[:, 1]     # [2, 5] (entire column 1)
    arr2d[0:2, 1:3] # Slice rows 0-1, columns 1-2
    ```
* **Boolean Indexing**:
    ```python
    arr = np.array([10, 20, 30, 40, 50])
    arr[arr > 30] # [40, 50] (elements greater than 30)
    ```

---

## 5. Array Manipulation

* **Reshaping**:
    ```python
    arr = np.arange(6) # [0, 1, 2, 3, 4, 5]
    arr.reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]
    arr.ravel()        # Flatten array to 1D
    arr.T              # Transpose (rows become columns, columns become rows)
    ```
* **Concatenating Arrays**:
    ```python
    a = np.array([1, 2])
    b = np.array([3, 4])
    np.concatenate((a, b)) # [1, 2, 3, 4]
    np.vstack((a, b))      # Stack vertically (rows)
    np.hstack((a, b))      # Stack horizontally (columns)
    ```
* **Splitting Arrays**:
    ```python
    arr = np.arange(9).reshape(3, 3)
    np.split(arr, 3, axis=0) # Split into 3 arrays along rows
    np.hsplit(arr, 3)        # Split horizontally (columns)
    ```

---

## 6. Universal Functions (ufuncs) & Operations

NumPy's power comes from its element-wise operations, which are highly optimized.

* **Arithmetic Operations**: Applied element-wise.
    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    a + b         # Element-wise addition: [5, 7, 9]
    a * 2         # Scalar multiplication: [2, 4, 6]
    np.add(a, b)  # Same as a + b
    ```
* **Comparison Operations**: Element-wise, returns boolean array.
    ```python
    a > 2         # [False, False, True]
    ```
* **Mathematical Functions**:
    ```python
    np.sqrt(a)    # Square root of each element
    np.sin(a)     # Sine of each element
    np.exp(a)     # Exponential of each element
    np.log(a)     # Natural logarithm
    ```
* **Aggregation Functions**:
    ```python
    arr.sum()      # Sum of all elements
    arr.mean()     # Mean of all elements
    arr.min()      # Minimum value
    arr.max()      # Maximum value
    arr.std()      # Standard deviation
    arr.sum(axis=0) # Sum along columns (for 2D array)
    arr.mean(axis=1) # Mean along rows (for 2D array)
    ```
* **Broadcasting**: NumPy's ability to perform operations on arrays of different shapes. It automatically "stretches" the smaller array to match the larger one.
    ```python
    a = np.array([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
    b = np.array([10, 20, 30])           # shape (3,)
    a + b  # b is broadcasted to (2, 3) for element-wise addition
    ```

---
with example and preferbly in pdf file
I can provide you with a comprehensive cheatsheet for **NumPy** and **Pandas** with examples, but I cannot directly generate or attach a PDF file. My responses are in text format.

However, you can easily copy and paste the content I provide into a document editor (like Google Docs, Microsoft Word, or a simple text editor), format it as you like, and then save or print it as a PDF. This way, you'll have your personalized cheatsheet in PDF format!

Here's the cheatsheet, complete with examples for each concept:

---

# 🐼 Pandas Cheatsheet: Data Analysis & Manipulation

Pandas is built on NumPy and provides powerful, flexible data structures for working with tabular and time-series data. It's essential for data cleaning, preparation, and analysis.

**Import Convention:** `import pandas as pd`

## 1. Data Structures

Pandas primarily uses two data structures:

* **`Series`**: A 1-dimensional labeled array. Think of it like a single column in a spreadsheet.
    ```python
    import pandas as pd
    s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'], name='My_Series')
    print(s)
    # Output:
    # a    10
    # b    20
    # c    30
    # d    40
    # Name: My_Series, dtype: int64
    ```
    

* **`DataFrame`**: A 2-dimensional labeled data structure with columns of potentially different types. It's like a spreadsheet or a SQL table.
    ```python
    data = {'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'Los Angeles', 'Chicago']}
    df = pd.DataFrame(data)
    print(df)
    # Output:
    #       Name  Age         City
    # 0    Alice   25     New York
    # 1      Bob   30  Los Angeles
    # 2  Charlie   35      Chicago
    ```
    

---

## 2. Creating DataFrames

* **From Dictionary**:
    ```python
    df_dict = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    print(df_dict)
    # Output:
    #    col1  col2
    # 0     1     4
    # 1     2     5
    # 2     3     6
    ```
* **From List of Lists**:
    ```python
    df_list = pd.DataFrame([[10, 20], [30, 40]], columns=['A', 'B'])
    print(df_list)
    # Output:
    #     A   B
    # 0  10  20
    # 1  30  40
    ```
* **Reading from Files**: (Assuming you have `data.csv` and `data.xlsx` files)
    ```python
    # df_csv = pd.read_csv('data.csv')
    # df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
    # print(df_csv.head())
    ```

---

## 3. Viewing & Inspecting Data

Let's use our `df` from above for these examples:
```python
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                   'Age': [25, 30, 35, 28, 42],
                   'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Houston'],
                   'Salary': [70000, 85000, 95000, 72000, 110000]})

print("df.head(2):\n", df.head(2))
# Output:
#       Name  Age         City  Salary
# 0    Alice   25     New York   70000
# 1      Bob   30  Los Angeles   85000

print("\ndf.tail(1):\n", df.tail(1))
# Output:
#    Name  Age     City    Salary
# 4  Eve   42  Houston  110000

print("\ndf.info():")
df.info()
# Output (partial):
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Name    5 non-null      object
#  1   Age     5 non-null      int64
#  2   City    5 non-null      object
#  3   Salary  5 non-null      int64
# dtypes: int64(2), object(2)
# memory usage: 288.0+ bytes

print("\ndf.describe():\n", df.describe())
# Output (partial):
#              Age        Salary
# count   5.000000      5.000000
# mean   32.000000  86400.000000
# std     6.745369  16522.711585
# min    25.000000  70000.000000
# 25%    28.000000  72000.000000
# 50%    30.000000  85000.000000
# 75%    35.000000  95000.000000
# max    42.000000 110000.000000

print("\ndf.shape:", df.shape) # (rows, columns)
# Output: df.shape: (5, 4)

print("\ndf.columns:", df.columns)
# Output: df.columns: Index(['Name', 'Age', 'City', 'Salary'], dtype='object')

print("\ndf['City'].value_counts():\n", df['City'].value_counts())
# Output:
# City
# New York       2
# Los Angeles    1
# Chicago        1
# Houston        1
# Name: count, dtype: int64

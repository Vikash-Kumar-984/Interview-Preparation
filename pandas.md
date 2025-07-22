

# 🐼 Comprehensive Pandas Cheatsheet: Data Analysis & Manipulation

Pandas is a powerful, open-source data analysis and manipulation library for Python. It provides flexible data structures and functions designed to make working with "relational" or "labeled" data easy and intuitive. It's built on top of NumPy.

**Import Convention:** `import pandas as pd`

## 1. Data Structures 🧱

Pandas revolves around two primary data structures:

### **`Series`**: 1-dimensional labeled array.
* **Analogy**: A single column of data in a spreadsheet, or a single vector.
* **Key Feature**: Has an `index` (labels for each row).

    ```python
    import pandas as pd
    s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'], name='My_Data')
    print(s)
    # Output:
    # a    10
    # b    20
    # c    30
    # d    40
    # Name: My_Data, dtype: int64
    ```
    

### **`DataFrame`**: 2-dimensional labeled data structure with columns of potentially different types.
* **Analogy**: A spreadsheet, a SQL table, or a dictionary of Series objects.
* **Key Features**: Both row and column indices.

    ```python
    data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 28],
            'City': ['NY', 'LA', 'Chicago', 'NY'],
            'Salary': [70000, 85000, 95000, 72000]}
    df = pd.DataFrame(data)
    print(df)
    # Output:
    #       Name  Age     City  Salary
    # 0    Alice   25       NY   70000
    # 1      Bob   30       LA   85000
    # 2  Charlie   35  Chicago   95000
    # 3    David   28       NY   72000
    ```
    

---

## 2. Creating DataFrames & Series ➕

### From Scratch:

* **From a Dictionary of Lists/Arrays (common for DataFrame):**
    ```python
    df_from_dict = pd.DataFrame({
        'ColA': [1, 2, 3],
        'ColB': ['X', 'Y', 'Z']
    })
    print(df_from_dict)
    ```
* **From a List of Dictionaries (each dict is a row):**
    ```python
    df_from_list_of_dicts = pd.DataFrame([
        {'col1': 1, 'col2': 'A'},
        {'col1': 2, 'col2': 'B'}
    ])
    print(df_from_list_of_dicts)
    ```
* **Creating an empty DataFrame:**
    ```python
    empty_df = pd.DataFrame(columns=['A', 'B'])
    ```
* **From a NumPy array (specify columns/index):**
    ```python
    import numpy as np
    df_from_np = pd.DataFrame(np.random.rand(3, 2), columns=['Col1', 'Col2'])
    print(df_from_np)
    ```

### Reading from Files:

* **CSV (Comma Separated Values):**
    ```python
    # df = pd.read_csv('your_file.csv')
    # Key arguments:
    #   sep=',': delimiter
    #   header=0: row number to use as column names (0-indexed)
    #   names=['col1', 'col2']: list of column names if no header
    #   index_col='ID': column to use as row index
    #   skiprows=[0, 2]: skip specific row numbers
    #   na_values=['N/A', 'None']: additional strings to recognize as NaN
    ```
* **Excel:**
    ```python
    # df = pd.read_excel('your_file.xlsx', sheet_name='Sheet1')
    ```
* **JSON:**
    ```python
    # df = pd.read_json('your_file.json')
    ```
* **SQL (using SQLAlchemy engine):**
    ```python
    # from sqlalchemy import create_engine
    # engine = create_engine('postgresql://user:pass@host:port/db')
    # df = pd.read_sql('SELECT * FROM my_table', engine)
    ```

---

## 3. Viewing & Inspecting Data 🔍

Let's use the `df` from section 1 for these examples:

* **First/Last N rows:**
    ```python
    print(df.head(2)) # Top 2 rows
    print(df.tail(1)) # Last 1 row
    ```
* **Summary Information (non-null counts, dtypes):**
    ```python
    df.info()
    ```
* **Descriptive Statistics (numerical columns):**
    ```python
    print(df.describe())
    print(df.describe(include='all')) # Include non-numerical columns
    ```
* **Shape (rows, columns):**
    ```python
    print(df.shape) # Output: (4, 4)
    ```
* **Column Names:**
    ```python
    print(df.columns) # Output: Index(['Name', 'Age', 'City', 'Salary'], dtype='object')
    ```
* **Row Index:**
    ```python
    print(df.index) # Output: RangeIndex(start=0, stop=4, step=1)
    ```
* **Data Types of Columns:**
    ```python
    print(df.dtypes)
    ```
* **Unique Values & Counts (for Series/column):**
    ```python
    print(df['City'].unique())        # Output: ['NY' 'LA' 'Chicago']
    print(df['City'].nunique())       # Output: 3 (number of unique cities)
    print(df['City'].value_counts())  # Counts occurrences of each unique value
    ```
* **Check for Null Values:**
    ```python
    print(df.isnull().sum()) # Count missing values per column
    ```

---

## 4. Selection & Indexing 🎯

Accessing specific rows, columns, or cells.

### Basic Selection:

* **Single Column (returns Series):**
    ```python
    print(df['Name'])
    ```
* **Multiple Columns (returns DataFrame):**
    ```python
    print(df[['Name', 'Salary']])
    ```

### Row/Column Selection by Label (`.loc[]`):

* **Select Row(s) by Label:**
    ```python
    df.loc[0]                 # Select row with index label 0
    df.loc[[0, 2]]            # Select rows with index labels 0 and 2
    ```
* **Select Row/Column by Label:**
    ```python
    df.loc[0, 'Age']          # Value at row label 0, column label 'Age' (Output: 25)
    df.loc[0:2, ['Name', 'City']] # Rows with labels 0, 1, 2 (inclusive), specific columns
    ```
* **Selecting all rows, specific columns by label:**
    ```python
    df.loc[:, 'City']         # All rows, 'City' column (returns Series)
    df.loc[:, ['City', 'Salary']] # All rows, 'City' and 'Salary' columns (returns DataFrame)
    ```

### Row/Column Selection by Integer Position (`.iloc[]`):

* **Select Row(s) by Position:**
    ```python
    df.iloc[0]                # Select row at integer position 0
    df.iloc[[0, 2]]           # Select rows at positions 0 and 2
    ```
* **Select Row/Column by Position:**
    ```python
    df.iloc[0, 1]             # Value at row position 0, column position 1 (Output: 25)
    df.iloc[0:3, 0:2]         # Slice rows from 0 up to (but not including) 3, columns 0 up to (but not including) 2
    ```
* **Selecting all rows, specific columns by position:**
    ```python
    df.iloc[:, 2]             # All rows, column at position 2 ('City')
    df.iloc[:, [2, 3]]        # All rows, columns at positions 2 and 3
    ```

### Boolean Indexing (Filtering Data):

* **Single Condition:**
    ```python
    # Select rows where Age > 30
    print(df[df['Age'] > 30])
    # Output:
    #       Name  Age     City  Salary
    # 2  Charlie   35  Chicago   95000
    ```
* **Multiple Conditions (AND - `&`, OR - `|`):**
    ```python
    # Select rows where Age is > 25 AND City is 'NY'
    print(df[(df['Age'] > 25) & (df['City'] == 'NY')])
    # Output:
    #     Name  Age City  Salary
    # 3  David   28   NY   72000

    # Select rows where City is 'NY' OR 'LA'
    print(df[df['City'].isin(['NY', 'LA'])])
    ```

---

## 5. Data Cleaning & Preparation 🧹

Dealing with messy data is a crucial part of data analysis.

### Handling Missing Values (NaN):

* **Detect Missing Values:**
    ```python
    df_missing = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    print(df_missing.isnull())     # Returns boolean DataFrame
    print(df_missing.isnull().sum()) # Count NaNs per column
    ```
* **Drop Rows/Columns with NaN:**
    ```python
    df_dropped_rows = df_missing.dropna(axis=0) # Drop rows with ANY NaN
    df_dropped_cols = df_missing.dropna(axis=1) # Drop columns with ANY NaN
    df_dropped_all = df_missing.dropna(how='all') # Drop row/col only if ALL values are NaN
    df_dropped_thresh = df_missing.dropna(thresh=2) # Keep rows with at least 2 non-NaN values
    # Use inplace=True to modify DataFrame directly: df.dropna(inplace=True)
    ```
* **Fill Missing Values:**
    ```python
    df_filled_zero = df_missing.fillna(0) # Fill all NaNs with 0
    df_filled_mean = df_missing['A'].fillna(df_missing['A'].mean()) # Fill column 'A' with its mean
    df_filled_ffill = df_missing.fillna(method='ffill') # Forward fill (use previous valid observation)
    df_filled_bfill = df_missing.fillna(method='bfill') # Backward fill (use next valid observation)
    df_filled_specific = df_missing.fillna({'A': 0, 'B': 99}) # Fill with different values for different columns
    ```

### Changing Data Types:

* **Using `astype()`:**
    ```python
    df['Age'] = df['Age'].astype(float) # Convert 'Age' to float
    df['Salary'] = df['Salary'].astype(str) # Convert 'Salary' to string
    # Common dtypes: int, float, object (string), bool, datetime64
    ```
* **Converting to Datetime:**
    ```python
    dates = pd.Series(['2023-01-01', '2023-01-02', 'Invalid Date'])
    df_dates = pd.to_datetime(dates, errors='coerce') # Convert to datetime, turn errors into NaT (Not a Time)
    print(df_dates)
    ```
* **Converting to Numeric (with errors handling):**
    ```python
    num_str = pd.Series(['1', '2', 'invalid', '4'])
    df_nums = pd.to_numeric(num_str, errors='coerce') # Convert to numeric, turn errors into NaN
    print(df_nums)
    ```

### Renaming Columns/Index:

* **Renaming specific columns:**
    ```python
    df_renamed = df.rename(columns={'Name': 'Full_Name', 'City': 'Location'})
    # Use inplace=True to modify original: df.rename(columns={'Name': 'Full_Name'}, inplace=True)
    ```
* **Renaming index labels:**
    ```python
    df_renamed_idx = df.rename(index={0: 'First', 1: 'Second'})
    ```

### Removing Duplicates:

* **Detect Duplicates:**
    ```python
    df_dup = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['X', 'Y', 'Y', 'Z']})
    print(df_dup.duplicated()) # Returns boolean Series indicating duplicate rows
    print(df_dup.duplicated(subset=['A'])) # Check duplicates based on 'A' column only
    ```
* **Drop Duplicate Rows:**
    ```python
    df_no_dup = df_dup.drop_duplicates() # Drops duplicate rows (keeps first occurrence by default)
    df_no_dup_subset = df_dup.drop_duplicates(subset=['A'], keep='last') # Drop based on 'A', keep last
    # Use inplace=True to modify original: df.drop_duplicates(inplace=True)
    ```

### Applying Functions (Row/Column-wise):

* **Using `apply()` on a Series (column):**
    ```python
    df['Name_Length'] = df['Name'].apply(len) # Calculate length of names
    print(df[['Name', 'Name_Length']])
    ```
* **Using `apply()` on a DataFrame (row or column-wise):**
    ```python
    # Apply a function to each row
    df['Bonus'] = df.apply(lambda row: row['Salary'] * 0.1 if row['Age'] > 30 else 0, axis=1)
    print(df[['Name', 'Salary', 'Bonus']])

    # Apply to each column
    df_norm = df[['Age', 'Salary']].apply(lambda x: (x - x.mean()) / x.std())
    print(df_norm)
    ```
* **Using `map()` for Series value substitution:**
    ```python
    city_map = {'NY': 'New York', 'LA': 'Los Angeles'}
    df['City_Full'] = df['City'].map(city_map)
    print(df[['City', 'City_Full']])
    ```

---

## 6. Data Manipulation & Transformation ⚙️

### Adding/Modifying Columns:

* **Adding a new column (constant value):**
    ```python
    df['Region'] = 'North'
    ```
* **Adding a new column (from existing columns):**
    ```python
    df['Salary_Per_Year'] = df['Salary'] / 12
    ```
* **Conditional Column Creation (`np.where` is efficient):**
    ```python
    df['Age_Group'] = np.where(df['Age'] >= 30, 'Adult', 'Young')
    print(df[['Age', 'Age_Group']])
    ```

### Dropping Columns/Rows:

* **Dropping columns:**
    ```python
    df_no_city = df.drop('City', axis=1) # Drop single column
    df_multiple_cols_dropped = df.drop(['Age', 'Salary'], axis=1) # Drop multiple columns
    # Use inplace=True for permanent modification
    ```
* **Dropping rows by index:**
    ```python
    df_no_first_row = df.drop(0, axis=0) # Drop row with index label 0
    df_no_specific_rows = df.drop([0, 2], axis=0) # Drop rows with index labels 0 and 2
    ```

### Sorting Data:

* **Sorting by column values:**
    ```python
    df_sorted_age = df.sort_values(by='Age', ascending=False) # Sort by Age, descending
    df_sorted_multi = df.sort_values(by=['City', 'Age'], ascending=[True, False]) # Sort by City (asc), then by Age (desc)
    ```
* **Sorting by index:**
    ```python
    df_sorted_index = df.sort_index(ascending=True)
    ```

### Grouping and Aggregating (`groupby`):

This is incredibly powerful for summarization.

* **Basic GroupBy (single aggregation):**
    ```python
    # Mean salary by city
    print(df.groupby('City')['Salary'].mean())
    # Output:
    # City
    # Chicago    95000.0
    # LA         85000.0
    # NY         71000.0
    # Name: Salary, dtype: float64
    ```
* **Multiple Aggregations:**
    ```python
    # Min, Max, and Count of Age by City
    print(df.groupby('City')['Age'].agg(['min', 'max', 'count']))
    # Output:
    #          min  max  count
    # City
    # Chicago   35   35      1
    # LA        30   30      1
    # NY        25   28      2
    ```
* **Applying different aggregations to different columns:**
    ```python
    print(df.groupby('City').agg(
        Avg_Age=('Age', 'mean'),
        Total_Salary=('Salary', 'sum'),
        Num_People=('Name', 'count')
    ))
    ```

### Merging, Joining, and Concatenating:

* **`pd.merge()` (for database-style joins):**
    * Combines DataFrames based on common columns (keys).
    ```python
    df_left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_left': [1, 2, 3]})
    df_right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value_right': [4, 5, 6]})

    # Inner join (default): only common keys
    merged_inner = pd.merge(df_left, df_right, on='key', how='inner')
    print("Inner Join:\n", merged_inner)
    # Output:
    #   key  value_left  value_right
    # 0   A           1            4
    # 1   B           2            5

    # Left join: all keys from left, matching from right (NaN if no match)
    merged_left = pd.merge(df_left, df_right, on='key', how='left')
    print("\nLeft Join:\n", merged_left)
    # Output:
    #   key  value_left  value_right
    # 0   A           1          4.0
    # 1   B           2          5.0
    # 2   C           3          NaN

    # Right join: all keys from right, matching from left
    merged_right = pd.merge(df_left, df_right, on='key', how='right')
    # Outer join: all keys from both, fill with NaN where no match
    merged_outer = pd.merge(df_left, df_right, on='key', how='outer')
    ```
* **`pd.concat()` (for stacking DataFrames):**
    * Stacks DataFrames along an axis (rows or columns).
    ```python
    df_top = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_bottom = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    df_concat_rows = pd.concat([df_top, df_bottom], axis=0) # Stack rows
    print("\nConcat Rows:\n", df_concat_rows)

    df_left_col = pd.DataFrame({'C': [9, 10]})
    df_concat_cols = pd.concat([df_top, df_left_col], axis=1) # Stack columns
    print("\nConcat Cols:\n", df_concat_cols)
    ```

### Pivoting and Melting (Reshaping):

* **`pivot_table()` (similar to Excel pivot tables):**
    * Creates a spreadsheet-style pivot table as a DataFrame.
    * Aggregates values based on rows, columns, and an aggregation function.
    ```python
    df_sales = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'Region': ['East', 'West', 'East', 'West'],
        'Sales': [100, 150, 120, 180]
    })
    pivot_table = df_sales.pivot_table(values='Sales', index='Date', columns='Region', aggfunc='sum')
    print("\nPivot Table:\n", pivot_table)
    # Output:
    # Region        East  West
    # Date
    # 2023-01-01   100   150
    # 2023-01-02   120   180
    ```
* **`melt()` (unpivoting from wide to long format):**
    * Transforms a DataFrame from wide format (multiple columns representing variables) to long format (fewer columns, where one column identifies the variable type and another holds the values).
    ```python
    df_wide = pd.DataFrame({
        'ID': ['A', 'B'],
        'Math': [90, 85],
        'Science': [75, 92]
    })
    df_long = pd.melt(df_wide, id_vars=['ID'], var_name='Subject', value_name='Score')
    print("\nMelted DataFrame:\n", df_long)
    # Output:
    #   ID  Subject  Score
    # 0  A     Math     90
    # 1  B     Math     85
    # 2  A  Science     75
    # 3  B  Science     92
    ```

---

# 📐 Comprehensive NumPy Cheatsheet: Numerical Operations

NumPy (Numerical Python) is the fundamental package for scientific computing with Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. It's the backbone of many other scientific libraries, including Pandas.

**Import Convention:** `import numpy as np`

## 1. NumPy Arrays (`ndarray`) 🌐

* **Core Concept**: A **NumPy array** is a grid of values, all of the **same type**, and is indexed by a tuple of non-negative integers. It's the most important object in NumPy.
* **Advantages over Python Lists**:
    * **Performance**: Much faster for numerical operations (implemented in C).
    * **Memory Efficiency**: Stores data in a contiguous block of memory.
    * **Functionality**: Offers a vast collection of high-level mathematical functions.
    

## 2. Array Creation ➕

### From Python Structures:

* **From List/Tuple:**
    ```python
    arr1d = np.array([1, 2, 3, 4, 5])
    arr2d = np.array([[10, 20, 30], [40, 50, 60]])
    print("1D Array:", arr1d)
    print("2D Array:\n", arr2d)
    ```
* **Specify Data Type:**
    ```python
    arr_float = np.array([1, 2, 3], dtype=np.float64)
    print("Float Array:", arr_float)
    ```

### Initial Placeholders:

* **Zeros:**
    ```python
    zeros_arr = np.zeros((3, 4)) # 3 rows, 4 columns
    print("Zeros Array:\n", zeros_arr)
    ```
* **Ones:**
    ```python
    ones_arr = np.ones((2, 2), dtype=int)
    print("Ones Array (int):\n", ones_arr)
    ```
* **Empty (uninitialized, fast):**
    ```python
    empty_arr = np.empty((2, 2)) # Content is random based on memory
    print("Empty Array:\n", empty_arr)
    ```
* **Full (filled with a constant value):**
    ```python
    full_arr = np.full((3, 3), 7) # 3x3 array filled with 7
    print("Full Array:\n", full_arr)
    ```
* **Identity Matrix:**
    ```python
    identity_mat = np.eye(3) # 3x3 identity matrix
    print("Identity Matrix:\n", identity_mat)
    ```

### Sequences & Ranges:

* **`arange` (like Python's `range`):**
    ```python
    arr_range = np.arange(0, 10, 2) # Start, Stop (exclusive), Step
    print("Arange (0 to 10, step 2):", arr_range) # Output: [0 2 4 6 8]
    ```
* **`linspace` (evenly spaced numbers):**
    ```python
    arr_linspace = np.linspace(0, 1, 5) # Start, Stop (inclusive), Number of elements
    print("Linspace (5 values from 0 to 1):", arr_linspace) # Output: [0.   0.25 0.5  0.75 1.  ]
    ```

### Random Numbers:

* **`np.random.rand(d0, d1, ...)`: Uniform distribution [0, 1) for given shape.**
    ```python
    rand_uniform = np.random.rand(2, 3) # 2 rows, 3 columns
    print("Random Uniform:\n", rand_uniform)
    ```
* **`np.random.randn(d0, d1, ...)`: Standard normal (Gaussian) distribution (mean=0, std=1).**
    ```python
    rand_normal = np.random.randn(2, 2)
    print("Random Normal:\n", rand_normal)
    ```
* **`np.random.randint(low, high, size)`: Random integers within a range.**
    ```python
    rand_int = np.random.randint(1, 10, size=(3, 2)) # Integers from 1 to 9, 3x2 array
    print("Random Integers:\n", rand_int)
    ```
* **`np.random.seed(seed)`: Set the seed for reproducibility.**
    ```python
    np.random.seed(42)
    print("Seeded random number:", np.random.rand())
    np.random.seed(42) # Resets the seed
    print("Same seeded random number:", np.random.rand())
    ```

---

## 3. Array Attributes ℹ️

Properties of `ndarray` objects.

```python
arr_attr_example = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

print("Shape (dimensions):", arr_attr_example.shape) # Output: (3, 3)
print("Number of dimensions:", arr_attr_example.ndim) # Output: 2
print("Total number of elements:", arr_attr_example.size) # Output: 9
print("Data type of elements:", arr_attr_example.dtype) # Output: int64
print("Size of each element in bytes:", arr_attr_example.itemsize) # Output: 8 (for int64)

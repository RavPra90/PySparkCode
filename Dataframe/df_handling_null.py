"""
COMPREHENSIVE GUIDE TO HANDLING NULLS IN PYSPARK
================================================

This guide covers everything from basic to advanced techniques for handling null values in PySpark,
with detailed explanations for beginners and production-level best practices.

Table of Contents:
1. Understanding Nulls in PySpark
2. Detecting and Counting Nulls
3. Filtering Data Based on Nulls
4. Replacing Nulls with Values
5. Handling Nulls in Aggregations
6. Advanced Null Handling Techniques
7. Advanced: Null-safe joins
8. Working with Nulls in Complex Data Types
9. Performance Considerations
10. Production Best Practices
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("PySpark Null Handling Guide") \
    .getOrCreate()

# Create sample data with nulls for demonstration
"""
SECTION 1: UNDERSTANDING NULLS IN PYSPARK
=========================================

Null values in PySpark represent missing or unknown data. They are different from:
- Empty strings ("")
- Zero values (0)
- Boolean false

IMPORTANT: In PySpark, nulls are represented as None in Python and null in Spark SQL.
When comparing nulls, standard comparisons don't work - you must use specific null-handling functions.
"""

# Creating a sample DataFrame with null values to demonstrate various techniques
data = [
    (1, "Alice", 28, 50000.0, datetime.date(2020, 1, 15)),
    (2, "Bob", None, 60000.0, None),
    (3, "Charlie", 35, None, datetime.date(2021, 5, 10)),
    (4, None, 42, 70000.0, datetime.date(2019, 3, 20)),
    (5, "Eve", None, None, None)
]

schema = StructType([
    StructField("id", IntegerType(), False),  # False means this field cannot be null
    StructField("name", StringType(), True),  # True means this field can be null
    StructField("age", IntegerType(), True),
    StructField("salary", DoubleType(), True),
    StructField("hire_date", DateType(), True)
])

df = spark.createDataFrame(data, schema)

# Display the DataFrame
print("Sample DataFrame:")
df.show()

"""
SECTION 2: DETECTING AND COUNTING NULLS
======================================

First step in handling nulls is to identify them in your dataset.
PySpark provides multiple ways to check for null values.
"""

# 2.1. Using isNull() and isNotNull() to check for null values
# TIP: These are the preferred methods for null checking, as they handle nulls correctly

print("\nFiltering for null names:")
df.filter(F.col("name").isNull()).show()

print("\nFiltering for non-null ages:")
df.filter(F.col("age").isNotNull()).show()

# 2.2. Count nulls in each column
# This is essential for data quality assessment in production pipelines

print("\nCounting nulls in each column:")
# Use a dynamic approach to count nulls in all columns
null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
null_counts.show()

# 2.3. Calculate percentage of nulls in each column
# PRODUCTION TIP: Set thresholds for acceptable null percentages in your data quality checks

print("\nPercentage of nulls in each column:")
total_rows = df.count()

# Create expressions to calculate null percentage for each column
null_percentages = []
for column_name in df.columns:
    null_percentage = (F.count(F.when(F.col(column_name).isNull(), column_name)) / total_rows * 100).alias(
        f"{column_name}_null_pct")
    null_percentages.append(null_percentage)

df.agg(*null_percentages).show()

# 2.4. Get rows with any null value
# Useful when you need complete records with no missing data

print("\nRows with any null value:")
# IMPORTANT: This creates a condition that checks if ANY column has a null
any_null_condition = F.lit(False)
for column_name in df.columns:
    any_null_condition = any_null_condition | F.col(column_name).isNull()

df.filter(any_null_condition).show()

# 2.5. Get rows with all complete values (no nulls)
# Often needed for clean analysis datasets

print("\nRows with no null values:")
# IMPORTANT: This creates a condition that checks if ALL columns are non-null
all_not_null_condition = F.lit(True)
for column_name in df.columns:
    all_not_null_condition = all_not_null_condition & F.col(column_name).isNotNull()

df.filter(all_not_null_condition).show()

"""
SECTION 3: FILTERING DATA BASED ON NULLS
======================================

Handling nulls often requires filtering the data to include or exclude records with nulls.
"""

# 3.1. Basic filtering examples

# Get employees with no salary information
print("\nEmployees with no salary information:")
df.filter(F.col("salary").isNull()).show()

# Get employees with complete age information
print("\nEmployees with age information:")
df.filter(F.col("age").isNotNull()).show()

# 3.2. Combining multiple null conditions
# IMPORTANT: Use proper boolean logic when combining conditions

print("\nEmployees with no age OR no hire date:")
df.filter(F.col("age").isNull() | F.col("hire_date").isNull()).show()

print("\nEmployees with both age AND salary:")
df.filter(F.col("age").isNotNull() & F.col("salary").isNotNull()).show()

# 3.3. Filtering rows where specific columns are all null or all non-null
# Useful for data completeness checks

selected_columns = ["age", "salary", "hire_date"]

# Rows where all selected columns are null
print("\nRows where age, salary, and hire_date are ALL null:")
all_null_condition = F.lit(True)
for col_name in selected_columns:
    all_null_condition = all_null_condition & F.col(col_name).isNull()
df.filter(all_null_condition).show()

# Rows where all selected columns have values
print("\nRows where age, salary, and hire_date ALL have values:")
all_not_null_condition = F.lit(True)
for col_name in selected_columns:
    all_not_null_condition = all_not_null_condition & F.col(col_name).isNotNull()
df.filter(all_not_null_condition).show()

"""
SECTION 4: REPLACING NULLS WITH VALUES
====================================

One of the most common tasks is to replace null values with something meaningful.
PySpark offers several approaches for this.
"""

# 4.1. Using fillna() to replace nulls with static values
# This is the simplest approach for basic null replacement

print("\nReplacing nulls with default values:")
# Replace nulls in all string columns with "Unknown"
df_filled_str = df.fillna("Unknown", subset=["name"])

# Replace nulls in all numeric columns with 0
df_filled_num = df_filled_str.fillna(0, subset=["age", "salary"])

# Replace nulls in date column with a specific date
today = datetime.date.today()
df_filled = df_filled_num.fillna(today, subset=["hire_date"])

df_filled.show()

# 4.2. Using fillna() with a dictionary to specify different values for different columns
# More flexible approach when different columns need different default values

print("\nReplacing nulls using a dictionary of values:")
fill_values = {
    "name": "Unknown",
    "age": 0,
    "salary": 0.0,
    "hire_date": datetime.date(1900, 1, 1)  # Using a sentinel date for missing dates
}
df_filled_dict = df.fillna(fill_values)
df_filled_dict.show()

# 4.3. Using when() and otherwise() for conditional replacement
# More powerful approach that allows different replacements based on conditions

print("\nConditional null replacement:")
# Replace age nulls with different values based on conditions
df_conditional = df.withColumn(
    "age",
    F.when(F.col("age").isNull() & (F.col("salary") > 55000), 30)  # If null age but high salary, assume 30
    .when(F.col("age").isNull(), 25)  # For other null ages, assume 25
    .otherwise(F.col("age"))  # Keep original value if not null
)
df_conditional.show()

# 4.4. Using na.fill() with different values for different types
# Efficient way to handle nulls across data types

print("\nFilling nulls by data type:")
# First with strings
df_by_type = df.na.fill("N/A", subset=["name"])
# Then with integers
df_by_type = df_by_type.na.fill(0, subset=["age"])
# Then with doubles
df_by_type = df_by_type.na.fill(0.0, subset=["salary"])
# Then with dates
df_by_type = df_by_type.na.fill(datetime.date(1900, 1, 1), subset=["hire_date"])

df_by_type.show()

# 4.5. Replacing nulls with statistical values
# PRODUCTION TIP: Often preferable to using static values for numeric columns

print("\nReplacing nulls with statistical values:")

# Calculate statistics first
age_stats = df.agg(
    F.mean("age").alias("mean_age"),
    F.median("age").alias("median_age"),
    F.min("age").alias("min_age"),
    F.max("age").alias("max_age")
).collect()[0]

salary_stats = df.agg(
    F.mean("salary").alias("mean_salary"),
    F.median("salary").alias("median_salary")
).collect()[0]

# Replace age nulls with mean age
df_stats = df.withColumn(
    "age",
    F.when(F.col("age").isNull(), age_stats["mean_age"])
    .otherwise(F.col("age"))
)

# Replace salary nulls with median salary (more robust to outliers than mean)
df_stats = df_stats.withColumn(
    "salary",
    F.when(F.col("salary").isNull(), salary_stats["median_salary"])
    .otherwise(F.col("salary"))
)

df_stats.show()

# 4.6. Using coalesce() to choose first non-null value
# Very useful when you have multiple potential sources for a value

print("\nUsing coalesce to find first non-null value:")
# Create a DataFrame with multiple potential sources for a value
employee_data = [
    (1, "John", None, "John Smith"),
    (2, None, "Jane Doe", None),
    (3, "Bob", "Robert Brown", None),
    (4, None, None, "Alice Johnson")
]
employee_df = spark.createDataFrame(employee_data, ["id", "first_name", "full_name", "display_name"])

# Use coalesce to pick the first non-null name available
employee_df_with_name = employee_df.withColumn(
    "best_name",
    F.coalesce(
        F.col("first_name"),
        F.col("full_name"),
        F.col("display_name"),
        F.lit("Unknown")  # Default if all are null
    )
)
employee_df_with_name.show()

"""
SECTION 5: HANDLING NULLS IN AGGREGATIONS
=======================================

Nulls can significantly impact aggregation results. PySpark provides functions to properly
handle nulls during aggregations.
"""

# Create a sample dataset for aggregation examples
agg_data = [
    ("Product A", 100, 15.0),
    ("Product B", None, 25.0),
    ("Product C", 200, None),
    ("Product D", None, None),
    ("Product A", 150, 20.0)
]
agg_df = spark.createDataFrame(agg_data, ["product", "quantity", "price"])

print("\nSample aggregation data:")
agg_df.show()

# 5.1. Basic aggregations and nulls
# IMPORTANT: Most aggregate functions ignore nulls by default

print("\nBasic aggregations with nulls:")
agg_df.groupBy("product").agg(
    F.count("quantity").alias("quantity_count"),  # Counts non-null values
    F.count("*").alias("row_count"),  # Counts all rows
    F.sum("quantity").alias("quantity_sum"),  # Sum ignores nulls
    F.avg("price").alias("avg_price")  # Average ignores nulls
).show()

# 5.2. Using expr() for more complex aggregation logic
# Allows SQL-like expressions for aggregation

print("\nComplex aggregations with expr():")
agg_df.groupBy("product").agg(
    F.expr("count(1)").alias("total_rows"),
    F.expr("count(quantity)").alias("valid_quantities"),
    F.expr("sum(case when quantity is not null and price is not null then quantity * price else 0 end)").alias(
        "total_value")
).show()

# 5.3. Using collect_list() and collect_set() with nulls
# These functions include nulls by default unless filtered

print("\nCollecting values with nulls:")
agg_df.groupBy("product").agg(
    F.collect_list("quantity").alias("all_quantities"),
    F.collect_list(F.when(F.col("quantity").isNotNull(), F.col("quantity"))).alias("non_null_quantities")
).show(truncate=False)

# 5.4. Custom null handling in aggregations
# For more control over how nulls affect your aggregations

print("\nCustom null handling in aggregations:")
agg_df.groupBy("product").agg(
    # Count of null quantities
    F.sum(F.when(F.col("quantity").isNull(), 1).otherwise(0)).alias("null_quantity_count"),

    # Sum treating nulls as zeros
    F.sum(F.coalesce(F.col("quantity"), F.lit(0))).alias("quantity_sum_nulls_as_zero"),

    # Average excluding products with null price
    F.when(
        F.count(F.when(F.col("price").isNotNull(), 1)) > 0,
        F.sum(F.coalesce(F.col("price"), F.lit(0))) / F.count(F.when(F.col("price").isNotNull(), 1))
    ).otherwise(0).alias("adjusted_avg_price")
).show()

"""
SECTION 6: DROPPING NULLS
======================

Before exploring more advanced techniques, let's cover how to properly drop nulls
from your DataFrame.
"""

# 6.1. Dropping rows with nulls
print("\nDropping rows with nulls:")

# Create a sample dataset with nulls in different positions
drop_data = [
    (1, "A", 10),
    (2, "B", None),
    (3, None, 30),
    (4, "D", 40),
    (5, None, None)
]
drop_df = spark.createDataFrame(drop_data, ["id", "name", "value"])

print("\nOriginal data:")
drop_df.show()

# Drop rows where ANY column contains null (default behavior)
print("\nDrop rows with ANY null (dropna()):")
drop_df.na.drop().show()

# Drop rows where ALL columns are null
print("\nDrop rows where ALL columns are null (dropna(how='all')):")
drop_df.na.drop(how="all").show()

# Drop rows where specific columns have nulls
print("\nDrop rows where 'name' is null (dropna(subset=['name'])):")
drop_df.na.drop(subset=["name"]).show()

# Drop rows where a minimum number of non-null values is not met
print("\nDrop rows with less than 2 non-null values (dropna(thresh=2)):")
drop_df.na.drop(thresh=2).show()

# 6.2. Dropping columns with too many nulls
# This isn't a built-in function, but we can implement it

print("\nDropping columns with too many nulls:")

# Create a larger sample dataset
many_nulls_data = [
    (1, "A", 10, 100, None),
    (2, "B", None, 200, 20),
    (3, None, 30, None, 30),
    (4, "D", 40, 400, 40),
    (5, None, None, 500, None)
]
many_nulls_df = spark.createDataFrame(many_nulls_data, ["id", "name", "value1", "value2", "value3"])

print("\nOriginal data:")
many_nulls_df.show()

# Calculate null percentage for each column
total_rows = many_nulls_df.count()
null_percentages = {}

for col_name in many_nulls_df.columns:
    null_count = many_nulls_df.filter(F.col(col_name).isNull()).count()
    null_percentage = (null_count / total_rows) * 100
    null_percentages[col_name] = null_percentage
    print(f"Column '{col_name}' has {null_percentage:.1f}% nulls")

# Define a threshold and get columns to keep
threshold = 40  # columns with null percentage > 40% will be dropped
columns_to_keep = [col for col, percentage in null_percentages.items() if percentage <= threshold]

print(f"\nKeeping columns with <= {threshold}% nulls: {columns_to_keep}")
reduced_df = many_nulls_df.select(columns_to_keep)
reduced_df.show()

# BEST PRACTICE: Always check the impact of dropping nulls on your dataset size
# This is especially important in production to avoid unexpected data loss
original_count = drop_df.count()
dropped_count = drop_df.na.drop().count()
dropped_percentage = ((original_count - dropped_count) / original_count) * 100

print(f"\nDropping nulls reduced the dataset by {dropped_percentage:.1f}%")
print(f"Original: {original_count} rows, After dropping nulls: {dropped_count} rows")

"""
SECTION 7: HANDLING NULLS IN JOINS
================================

Nulls in join columns can lead to unexpected results. Understanding how different
join types handle nulls is crucial for correct data processing.
"""

# 6.1. Window functions with null handling
# Useful for imputing missing values based on similar records

print("\nImputing nulls using window functions:")

# Create sample data with sequential values and some nulls
window_data = [
    (1, "A", 10.0),
    (2, "A", None),
    (3, "A", 30.0),
    (4, "A", 40.0),
    (5, "A", None),
    (6, "A", None),
    (7, "A", 70.0),
    (1, "B", None),
    (2, "B", 20.0),
    (3, "B", 30.0),
    (4, "B", None),
    (5, "B", 50.0)
]
window_df = spark.createDataFrame(window_data, ["id", "group", "value"])

print("\nOriginal window data:")
window_df.orderBy("group", "id").show()

# Import Window functions
from pyspark.sql import Window

# Define a window specification partitioned by group and ordered by id
window_spec = Window.partitionBy("group").orderBy("id")

# Method 1: Forward Fill (LOCF - Last Observation Carried Forward)
window_df_ffill = window_df.withColumn(
    "value_ffill",
    F.last(F.col("value"), ignorenulls=True).over(window_spec)
)

# Method 2: Backward Fill (NOCB - Next Observation Carried Backward)
# For this, we need to order in reverse, get last non-null, then restore original order
window_spec_reverse = Window.partitionBy("group").orderBy(F.desc("id"))
window_df_bfill = window_df_ffill.withColumn(
    "value_bfill",
    F.last(F.col("value"), ignorenulls=True).over(window_spec_reverse)
)

# Method 3: Linear interpolation (simplified)
# This is an approximation using the average of previous and next non-null values
window_df_with_neighbors = window_df.withColumn(
    "prev_value",
    F.lag(F.col("value")).over(window_spec)
).withColumn(
    "next_value",
    F.lead(F.col("value")).over(window_spec)
)

window_df_interp = window_df_with_neighbors.withColumn(
    "value_interp",
    F.when(F.col("value").isNull() & F.col("prev_value").isNotNull() & F.col("next_value").isNotNull(),
           (F.col("prev_value") + F.col("next_value")) / 2)
    .otherwise(F.col("value"))
)

print("\nData with forward fill, backward fill, and interpolation:")
window_df_interp.select(
    "id", "group", "value", "value_ffill", "value_bfill", "value_interp"
).orderBy("group", "id").show()

# 6.2. Drop rows or columns with too many nulls
# Useful for cleaning datasets before analysis

print("\nDropping rows with too many nulls:")

# Create a sample DataFrame with varying numbers of nulls
null_counts_data = [
    (1, "Alice", 25, None, None),  # 2 nulls
    (2, None, None, None, "NY"),  # 3 nulls
    (3, "Charlie", None, None, None),  # 3 nulls
    (4, "David", 35, 70000, "CA"),  # 0 nulls
    (5, None, 28, 55000, None)  # 2 nulls
]

null_df = spark.createDataFrame(null_counts_data, ["id", "name", "age", "salary", "state"])

# Drop rows where more than 2 fields are null
columns_to_check = ["name", "age", "salary", "state"]  # Exclude ID which should never be null

# Count nulls in each row for specified columns
null_df_with_counts = null_df.withColumn(
    "null_count",
    sum(F.when(F.col(c).isNull(), 1).otherwise(0) for c in columns_to_check)
)

print("\nRows with null counts:")
null_df_with_counts.show()

# Keep only rows with 2 or fewer nulls
clean_df = null_df_with_counts.filter(F.col("null_count") <= 2).drop("null_count")

print("\nRows with 2 or fewer nulls:")
clean_df.show()

# 6.3. Identifying columns with too many nulls
# Useful for feature selection in machine learning pipelines

print("\nIdentifying columns with high null percentages:")

# Calculate percentage of nulls in each column
total_rows = null_df.count()
null_percentages = []

for column in null_df.columns:
    null_count = null_df.filter(F.col(column).isNull()).count()
    null_percentage = (null_count / total_rows) * 100
    null_percentages.append((column, null_percentage))

# Convert to DataFrame for better display
null_percent_df = spark.createDataFrame(null_percentages, ["column", "null_percentage"])
null_percent_df.show()

print("\nColumns with more than 40% nulls (candidates for removal):")
null_percent_df.filter(F.col("null_percentage") > 40).show()

# 6.4. Using ML techniques to impute nulls
# For advanced use cases where statistical imputation is needed

print("\nUsing Spark ML for null imputation:")

# This requires Spark ML - import necessary libraries
from pyspark.ml.feature import Imputer

# Prepare data (select only numeric columns for imputation)
impute_df = df.select("id", "age", "salary")

# Create an imputer
imputer = Imputer(
    inputCols=["age", "salary"],
    outputCols=["age_imputed", "salary_imputed"],
    strategy="mean"  # Could also use "median" or "mode"
)

# Fit the imputer and transform the data
imputer_model = imputer.fit(impute_df)
imputed_df = imputer_model.transform(impute_df)

print("\nData with imputed values:")
imputed_df.show()

# ----------------------------------------------------------------------------------------------------------------------
# 7. Advanced: Null-safe joins
#   - Spark's default join treats null != null, so rows with null keys won't match.
#   - Use eqNullSafe() for null-safe equality: null <=> null is true.
# ----------------------------------------------------------------------------------------------------------------------

# 7.1. Creating sample datasets with nulls for join examples

print("\nCreating datasets for join examples:")

# Left dataset - Employee data with some nulls in department_id
left_data = [
    (1, "Alice", 101),
    (2, "Bob", None),  # Null department_id
    (3, "Charlie", 103),
    (4, "David", 102),
    (5, "Eve", None),  # Null department_id
    (6, "Frank", 106)  # No matching department
]
left_df = spark.createDataFrame(left_data, ["emp_id", "emp_name", "department_id"])

# Right dataset - Department data with some nulls
right_data = [
    (101, "HR", "Building A"),
    (102, "Finance", "Building B"),
    (103, "IT", "Building C"),
    (104, "Marketing", None),  # Null location
    (None, "Unknown", "Building Z")  # Null department_id
]
right_df = spark.createDataFrame(right_data, ["department_id", "dept_name", "location"])

print("\nEmployee data (left_df):")
left_df.show()

print("\nDepartment data (right_df):")
right_df.show()

# 7.2. Understanding how different join types handle nulls

# Inner join - Nulls don't match
print("\nINNER JOIN - Nulls don't match other nulls:")
inner_join = left_df.join(right_df, "department_id", "inner")
inner_join.show()

# Left join - Keep all rows from left, nulls don't match
print("\nLEFT JOIN - Keep all left rows, nulls don't match:")
left_join = left_df.join(right_df, "department_id", "left")
left_join.show()

# Right join - Keep all rows from right, nulls don't match
print("\nRIGHT JOIN - Keep all right rows, nulls don't match:")
right_join = left_df.join(right_df, "department_id", "right")
right_join.show()

# Full outer join - Keep all rows, nulls don't match
print("\nFULL OUTER JOIN - Keep all rows, nulls don't match:")
full_join = left_df.join(right_df, "department_id", "full")
full_join.show()

# 7.3. Null-safe joins
# IMPORTANT: Use the <=> operator (null-safe equals) to match null values with other nulls

print("\nNULL-SAFE JOIN - Using <=> to match nulls with nulls:")
null_safe_join = left_df.join(
    right_df,
    left_df["department_id"].eqNullSafe(right_df["department_id"]),  # Proper null-safe comparison
    "inner"
)
null_safe_join.show()

# Alternative approach using expr
print("\nNULL-SAFE JOIN - Using expr approach:")
null_safe_join_expr = left_df.alias("l").join(
    right_df.alias("r"),
    F.expr("l.department_id <=> r.department_id"),
    "inner"
)
null_safe_join_expr.show()

# 7.4. Pre-processing nulls before joins
# Often a practical approach in production

print("\nPre-processing nulls before joining:")

# Replace nulls with a sentinel value
left_df_prep = left_df.withColumn(
    "department_id",
    F.when(F.col("department_id").isNull(), -999).otherwise(F.col("department_id"))
)

right_df_prep = right_df.withColumn(
    "department_id",
    F.when(F.col("department_id").isNull(), -999).otherwise(F.col("department_id"))
)

# Now join - nulls will match since they're all replaced with -999
prep_join = left_df_prep.join(right_df_prep, "department_id", "inner")

print("\nJoin after replacing nulls with sentinel value (-999):")
prep_join.show()

# 7.5. Handling nulls in non-equi joins
print("\nHandling nulls in non-equi joins:")

# Create datasets for range join example
range_left = spark.createDataFrame([
    (1, "A", 10, 20),
    (2, "B", None, 40),  # Null min_value
    (3, "C", 30, None),  # Null max_value
    (4, "D", 40, 50)
], ["id", "name", "min_value", "max_value"])

range_right = spark.createDataFrame([
    (101, "X", 15),
    (102, "Y", 35),
    (103, "Z", None)  # Null value
], ["id", "code", "value"])

print("\nRange join data:")
print("Left (ranges):")
range_left.show()
print("Right (values):")
range_right.show()

# Non-equi join with null handling
range_join = range_left.join(
    range_right,
    (range_right["value"] >= range_left["min_value"]) &
    (range_right["value"] <= range_left["max_value"]),
    "inner"
)

print("\nRange join result (nulls cause no matches):")
range_join.show()

# Handle nulls before non-equi join
range_left_handled = range_left.filter(
    F.col("min_value").isNotNull() & F.col("max_value").isNotNull()
)

range_right_handled = range_right.filter(F.col("value").isNotNull())

range_join_handled = range_left_handled.join(
    range_right_handled,
    (range_right_handled["value"] >= range_left_handled["min_value"]) &
    (range_right_handled["value"] <= range_left_handled["max_value"]),
    "inner"
)

print("\nRange join after handling nulls:")
range_join_handled.show()

# 7.6. Best practices for handling nulls in joins

# BEST PRACTICE 1: Always analyze null distribution in join columns before joining
print("\nAnalyzing null distribution in join columns:")
left_null_count = left_df.filter(F.col("department_id").isNull()).count()
left_total = left_df.count()
print(
    f"Left join column has {left_null_count} nulls out of {left_total} rows ({left_null_count / left_total * 100:.1f}%)")

right_null_count = right_df.filter(F.col("department_id").isNull()).count()
right_total = right_df.count()
print(
    f"Right join column has {right_null_count} nulls out of {right_total} rows ({right_null_count / right_total * 100:.1f}%)")

# BEST PRACTICE 2: Check row counts before and after joins to detect unexpected losses
print("\nChecking row counts before and after joins:")
left_count = left_df.count()
right_count = right_df.count()
joined_count = inner_join.count()

print(f"Left rows: {left_count}, Right rows: {right_count}, Inner joined rows: {joined_count}")
# Calculate the theoretical maximum (cartesian product) minus null matches
theoretical_max = (left_count - left_null_count) * (right_count - right_null_count)
print(f"Theoretical maximum matches (without nulls): {theoretical_max}")

# BEST PRACTICE 3: Document join null handling strategy in production code
"""
# In production code, document your null handling strategy for joins:
#
# 1. NULL HANDLING STRATEGY: REPLACE
#    - Replace nulls in join columns with sentinel values before joining
#    - Example: df = df.withColumn("key", F.coalesce(F.col("key"), F.lit(-999)))
#
# 2. NULL HANDLING STRATEGY: FILTER
#    - Filter out rows with nulls in join columns before joining
#    - Example: df = df.filter(F.col("key").isNotNull())
#
# 3. NULL HANDLING STRATEGY: NULL-SAFE
#    - Use null-safe equals operator (<=>)
#    - Example: df1.join(df2, F.col("df1.key") <=> F.col("df2.key"), "inner")
#
# 4. NULL HANDLING STRATEGY: SPECIAL CASE
#    - Handle nulls as a special business case with custom logic
#    - Example: Create separate DataFrames for null and non-null cases,
#      process them differently, then union results
"""


# BEST PRACTICE 4: Create a reusable function for consistent null handling in joins
def prepare_for_join(dataframe, join_columns, strategy="replace", sentinel_value=-999):
    """
    Prepare a DataFrame's join columns for consistent null handling.

    Args:
        dataframe: The DataFrame to prepare
        join_columns: List of column names used for joining
        strategy: One of "replace", "filter", or "none"
        sentinel_value: Value to use when replacing nulls

    Returns:
        Prepared DataFrame
    """
    result_df = dataframe

    if strategy == "replace":
        for col in join_columns:
            result_df = result_df.withColumn(
                col,
                F.when(F.col(col).isNull(), sentinel_value).otherwise(F.col(col))
            )
    elif strategy == "filter":
        filter_condition = F.lit(True)
        for col in join_columns:
            filter_condition = filter_condition & F.col(col).isNotNull()
        result_df = result_df.filter(filter_condition)
    elif strategy != "none":
        raise ValueError("Strategy must be one of: replace, filter, none")

    return result_df


# Example usage:
print("\nUsing a reusable function for consistent null handling:")
prepared_left = prepare_for_join(left_df, ["department_id"], "replace")
prepared_right = prepare_for_join(right_df, ["department_id"], "replace")
consistent_join = prepared_left.join(prepared_right, "department_id", "inner")
consistent_join.show()

"""
SECTION 8: WORKING WITH NULLS IN COMPLEX DATA TYPES
================================================

Handling nulls becomes more complex with nested structures like arrays, maps, and structs.
"""

# 8.1. Nulls in Array columns
array_data = [
    (1, ["apple", "orange", None, "banana"]),
    (2, None),
    (3, ["grape", None]),
    (4, [])
]
array_df = spark.createDataFrame(array_data, ["id", "fruits"])

print("\nHandling nulls in arrays:")
array_df.show(truncate=False)

# Filter out null elements from arrays
array_df_clean = array_df.withColumn(
    "fruits_no_nulls",
    F.expr("filter(fruits, x -> x IS NOT NULL)")
)

# Replace the entire array if it's null
array_df_clean = array_df_clean.withColumn(
    "fruits_no_nulls",
    F.when(F.col("fruits").isNull(), F.array()).otherwise(F.col("fruits_no_nulls"))
)

print("\nArrays with nulls removed:")
array_df_clean.show(truncate=False)

# Count non-null elements in arrays
array_df_clean = array_df_clean.withColumn(
    "non_null_count",
    F.expr("size(filter(fruits, x -> x IS NOT NULL))")
)

print("\nCounting non-null elements in arrays:")
array_df_clean.show(truncate=False)

# 8.2. Nulls in Map columns
map_data = [
    (1, {"name": "John", "age": "30", "city": None}),
    (2, {"name": "Mary", "age": None}),
    (3, None),
    (4, {})
]
map_df = spark.createDataFrame(map_data, ["id", "attributes"])

print("\nHandling nulls in maps:")
map_df.show(truncate=False)

# Check if map itself is null and replace with empty map
map_df_clean = map_df.withColumn(
    "attributes",
    F.when(F.col("attributes").isNull(), F.create_map()).otherwise(F.col("attributes"))
)

# Check for specific key and provide default if null
map_df_clean = map_df_clean.withColumn(
    "name",
    F.when(F.col("attributes")["name"].isNull(), "Unknown").otherwise(F.col("attributes")["name"])
)

map_df_clean = map_df_clean.withColumn(
    "age",
    F.when(F.col("attributes")["age"].isNull(), "0").otherwise(F.col("attributes")["age"])
)

print("\nExtracted map values with defaults for nulls:")
map_df_clean.show(truncate=False)

# 7.3. Nulls in Struct columns
struct_data = [
    (1, {"first": "John", "last": "Doe"}),
    (2, {"first": "Jane", "last": None}),
    (3, {"first": None, "last": "Smith"}),
    (4, None)
]

# Define schema for struct
struct_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StructType([
        StructField("first", StringType(), True),
        StructField("last", StringType(), True)
    ]), True)
])

struct_df = spark.createDataFrame(struct_data, struct_schema)

print("\nHandling nulls in structs:")
struct_df.show(truncate=False)

# Check if struct itself is null
struct_df_clean = struct_df.withColumn(
    "has_name",
    F.col("name").isNotNull()
)

# Access and handle null fields within struct
struct_df_clean = struct_df_clean.withColumn(
    "first_name",
    F.when(F.col("name").isNull(), "Unknown")
    .when(F.col("name.first").isNull(), "Unknown")
    .otherwise(F.col("name.first"))
)

struct_df_clean = struct_df_clean.withColumn(
    "last_name",
    F.when(F.col("name").isNull(), "Unknown")
    .when(F.col("name.last").isNull(), "Unknown")
    .otherwise(F.col("name.last"))
)

print("\nExtracted struct values with defaults for nulls:")
struct_df_clean.show(truncate=False)

"""
SECTION 9: PERFORMANCE CONSIDERATIONS
===================================

Handling nulls efficiently is crucial for performance in big data applications.
"""

# 9.1. Performance Tips for Null Handling

print("\nPerformance considerations for null handling:")

# PERFORMANCE TIP 1: Use isNull()/isNotNull() instead of equality comparisons
# Correct way:
# df.filter(F.col("age").isNull())

# Incorrect way (will not work as expected):
# df.filter(F.col("age") == None)

# PERFORMANCE TIP 2: For multiple null checks, combine them efficiently
# More efficient for filter pushdown:
# df.filter(F.col("age").isNull() | F.col("salary").isNull())

# Less efficient (evaluates each filter separately):
# df.filter(F.col("age").isNull()).filter(F.col("salary").isNull())

# PERFORMANCE TIP 3: For complex datasets, handle nulls as early as possible
# Often more efficient to handle nulls early in the pipeline
# df_clean = df.na.fill(default_values)
# ... rest of pipeline with clean data ...

# PERFORMANCE TIP 4: Use when/otherwise for complex conditional logic rather than
# multiple withColumn operations
# More efficient:
# df.withColumn("age",
#     F.when(F.col("age").isNull(), default_age)
#     .otherwise(F.col("age"))
# )

# Less efficient:
# df_tmp = df.withColumn("age",
#     F.when(F.col("age").isNull(), default_age)
#     .otherwise(F.col("age"))
# )
# df_tmp.withColumn(...) # additional operations

# PERFORMANCE TIP 5: For statistical imputations, cache intermediate results if reused
# stats_df = df.agg(...) # calculate stats once
# stats_df.cache() # cache if used multiple times
# values = stats_df.collect()[0]
# ... use values for imputation ...

"""
SECTION 10: PRODUCTION BEST PRACTICES
==================================

Best practices for handling nulls in production PySpark applications.
"""

# 10.1. Production Best Practices

print("\nProduction best practices for null handling:")


# BEST PRACTICE 1: Document null handling strategy for each column
# For production code, maintain a data dictionary that defines:
# - Which columns can have nulls
# - The meaning of nulls in each context
# - The chosen strategy for handling nulls

# BEST PRACTICE 2: Implement data quality checks to track null percentages
def track_null_percentages(dataframe, threshold=5.0):
    """
    Track null percentages in each column and raise warnings if above threshold.

    Args:
        dataframe: The DataFrame to check
        threshold: Maximum acceptable null percentage

    Returns:
        Dictionary of columns exceeding the threshold
    """
    results = {}
    total_rows = dataframe.count()

    for column in dataframe.columns:
        null_count = dataframe.filter(F.col(column).isNull()).count()
        null_percentage = (null_count / total_rows) * 100

        if null_percentage > threshold:
            results[column] = null_percentage

    return results


# Example usage:
# problem_columns = track_null_percentages(df, threshold=10.0)
# if problem_columns:
#     print(f"WARNING: Columns exceeding null threshold: {problem_columns}")
#     # In production, send alerts or log warnings

# BEST PRACTICE 3: For ETL jobs, create both cleaned and raw versions of data
# - Raw version: original data with nulls preserved
# - Cleaned version: nulls handled according to business rules
# This approach maintains data lineage and allows reprocessing if needed

# Example pseudocode:
# raw_df = spark.read.parquet("raw_data_path")
# raw_df.write.parquet("raw_data_archive_path")
#
# # Apply null handling and cleaning rules
# clean_df = apply_null_handling(raw_df)
# clean_df.write.parquet("clean_data_path")

# BEST PRACTICE 4: For mission-critical columns, implement null validation
def validate_no_nulls(dataframe, critical_columns):
    """
    Validate that critical columns have no null values.

    Args:
        dataframe: The DataFrame to check
        critical_columns: List of columns that must not have nulls

    Returns:
        True if validation passes, False otherwise
    """
    for column in critical_columns:
        if dataframe.filter(F.col(column).isNull()).count() > 0:
            return False
    return True


# Example usage:
# critical_cols = ["id", "transaction_date", "amount"]
# if not validate_no_nulls(df, critical_cols):
#     raise ValueError("Critical columns contain null values")

# BEST PRACTICE 5: For machine learning pipelines, handle nulls consistently
# between training and inference
# - Document null handling strategies
# - Implement as reusable functions
# - Apply the same strategy during both training and inference

# Example pseudocode:
# def prepare_features(dataframe):
#     """Apply consistent null handling for features."""
#     return dataframe.na.fill(fill_values)
#
# # Training
# train_df = prepare_features(raw_train_df)
# model = train_model(train_df)
#
# # Inference
# inference_df = prepare_features(raw_inference_df)
# predictions = model.transform(inference_df)

"""
CONCLUSION
==========

This guide covered comprehensive techniques for handling null values in PySpark.
From basic detection to advanced imputation strategies, these approaches will help
ensure your data processing pipelines handle missing data effectively.

Remember these key takeaways:
1. Always be aware of nulls in your data
2. Use isNull() and isNotNull() for proper null comparisons
3. Choose appropriate null handling strategies based on your data context
4. Consider the impact of nulls on aggregations and joins
5. Use the appropriate drop strategy when removing nulls
6. Remember that nulls don't match nulls in standard joins - use null-safe joins when needed
7. Document your null handling approach for production workflows
8. Test your code with edge cases containing various null patterns
9. Balance performance with correctness when handling nulls at scale

Happy coding with PySpark!
"""
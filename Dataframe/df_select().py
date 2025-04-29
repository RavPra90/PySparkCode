from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Create a SparkSession
spark = SparkSession.builder.appName("PySpark Select Tutorial").getOrCreate()

# ==========================================================================
# INTRODUCTION TO DATAFRAME SELECT
# ==========================================================================
# The `select` operation is one of the most fundamental operations in PySpark.
# It allows you to choose which columns to include in your result DataFrame,
# similar to SQL's SELECT statement.
#
# Let's create a sample DataFrame to demonstrate various select operations:

# Define schema
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), True),
    StructField("salary", DoubleType(), True),
    StructField("department", StringType(), True)
])

# Sample data
data = [
    (1, "John", 30, 50000.0, "Engineering"),
    (2, "Alice", 25, 45000.0, "Marketing"),
    (3, "Bob", None, 60000.0, "Engineering"),
    (4, "Sarah", 35, None, "HR"),
    (5, "Mike", 40, 70000.0, None)
]

# Create DataFrame
df = spark.createDataFrame(data, schema)

# Display schema and data
print("DataFrame Schema:")
df.printSchema()
print("DataFrame Data:")
df.show()

# ==========================================================================
# BASIC SELECT OPERATIONS
# ==========================================================================

# 1. Select specific columns by name
# ==========================================================================
# The simplest form of select is choosing columns by their name.
# SYNTAX: df.select("column1", "column2", ...)
# This returns a new DataFrame with only the specified columns.

print("1. Selecting specific columns:")
df.select("id", "name").show()

# Select the "age" column only
df.select("age").show()              # By column name strings
df.select(df.age).show()             # Using DataFrame attribute
df.select(F.col("salary")).show()      # Using col() function

# 2. Select all columns
# ==========================================================================
# You can select all columns using '*' syntax, similar to SQL
# PITFALL: This returns a new DataFrame but doesn't make a deep copy of the data
# BEST PRACTICE: Only use when you actually need all columns

print("2. Selecting all columns:")
df.select("*").show()

# Equivalent to '*' using the DataFrame's column list
df.select(*df.columns).show()   # same effect as select("*")

# 3. Select with column objects
# ==========================================================================
# Instead of string column names, you can use Column objects for more flexibility
# SYNTAX: df.select(df["column1"], df["column2"], ...)
# NOTE: This approach is required when you need to perform operations on columns

print("3. Selecting with column objects:")
df.select(df["id"], df["name"], df["salary"]).show()

# 4. Combined string and column object approaches
# ==========================================================================
# You can mix string column references and Column objects
# PITFALL: Mixing styles can make code less readable
# BEST PRACTICE: Be consistent with one approach throughout your code

print("4. Mixed approach:")
df.select("id", df["name"], "department").show()

# ==========================================================================
# TRANSFORMING COLUMNS DURING SELECT
# ==========================================================================

# 5. Renaming columns during select
# ==========================================================================
# You can rename columns during selection using the alias() method
# SYNTAX: df.select(df["column1"].alias("new_name"), ...)
# BEST PRACTICE: Use meaningful column names that indicate the transformation

print("5. Renaming columns during select:")
df.select(
    df["id"],
    df["name"].alias("full_name"),
    df["salary"].alias("annual_salary")
).show()

# 6. Applying functions to columns during select
# ==========================================================================
# You can transform column values using functions
# PRODUCTION TIP: This is more efficient than selecting first and transforming later
# since Spark can optimize the entire operation at once

print("6. Applying functions during select:")
df.select(
    df["id"],
    df["name"],
    (df["salary"] * 0.1).alias("bonus"),
    (F.upper(df["department"])).alias("DEPARTMENT")
).show()

# 7. Using PySpark SQL functions
# ==========================================================================
# PySpark provides many built-in functions through the functions module (imported as F)
# BEST PRACTICE: For complex transformations, SQL functions are often more readable
# and optimized than UDFs (User Defined Functions)

print("7. Using SQL functions in select:")
df.select(
    "id",
    F.concat(F.lit("Name: "), df["name"]).alias("employee"),
    F.when(df["age"].isNull(), F.lit("Unknown")).otherwise(df["age"]).alias("employee_age"),
    F.round(df["salary"], 2).alias("rounded_salary")
).show(truncate=False)

# ==========================================================================
# HANDLING NULL VALUES AND TYPE CONVERSIONS
# ==========================================================================

# 8. Handling NULL values during select
# ==========================================================================
# NULL values require special consideration in Spark
# PITFALL: Operations on NULL values generally result in NULL
# BEST PRACTICE: Use coalesce, isNull/isNotNull, or when/otherwise for NULL handling

print("8. Handling NULL values:")
df.select(
    "id",
    "name",
    # Replace NULL age with default value
    F.coalesce(df["age"], F.lit(0)).alias("age_or_zero"),
    # Check if salary is NULL
    df["salary"].isNull().alias("is_salary_missing"),
    # Conditional logic for department
    F.when(df["department"].isNull(), "Unassigned")
    .otherwise(df["department"]).alias("dept")
).show()

# 9. Type conversions during select
# ==========================================================================
# You can convert between data types during selection
# PITFALL: Failed conversions might result in NULL values
# PRODUCTION TIP: Always verify data quality before type conversion

print("9. Type conversions:")
df.select(
    "id",
    # Convert age to string
    df["age"].cast("string").alias("age_string"),
    # Convert salary to integer (truncates decimal portion)
    df["salary"].cast(IntegerType()).alias("salary_int"),
    # Multiple conversions
    F.concat(
        F.lit("ID:"),
        df["id"].cast("string")
    ).alias("id_label")
).show()

# ==========================================================================
# ADVANCED SELECT OPERATIONS
# ==========================================================================

# 10. Select with expressions
# ==========================================================================
# You can use SQL expressions directly in select
# PRODUCTION TIP: This can make code more readable when doing complex operations
# PITFALL: The expression syntax might be less IDE-friendly (no code completion)

print("10. Select with SQL expressions:")
df.selectExpr(
    "id",
    "name",
    "age",
    "CASE WHEN salary IS NULL THEN 0 ELSE salary * 1.1 END as adjusted_salary",
    "UPPER(department) as department"
).show()

# 11. Dynamic column selection
# ==========================================================================
# Sometimes you need to select columns dynamically based on conditions
# PRODUCTION TIP: This is useful for generic data processing pipelines

print("11. Dynamic column selection:")
# Select only string columns
string_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
df.select(*string_columns).show()

# Select only numeric columns
numeric_columns = [field.name for field in df.schema.fields
                   if isinstance(field.dataType, IntegerType) or isinstance(field.dataType, DoubleType)]
df.select(*numeric_columns).show()

# 12. Select with regular expressions
# ==========================================================================
# You can select columns that match a pattern
# PRODUCTION TIP: Useful when working with datasets that have many similarly named columns

print("12. Select with regular expressions:")
# Select all columns starting with 's'
df.select(df.colRegex("`s.*`")).show()

# 13. Select except certain columns
# ==========================================================================
# Sometimes it's easier to specify which columns to exclude
# PRODUCTION TIP: Useful when you have many columns but only want to exclude a few

print("13. Select except certain columns:")
columns_to_drop = ["salary", "department"]
remaining_columns = [col for col in df.columns if col not in columns_to_drop]
df.select(*remaining_columns).show()

# 14. Select with column position
# ==========================================================================
# You can select columns by their position/index
# PITFALL: Position-based selection can break if schema changes
# PRODUCTION TIP: Avoid in production code unless absolutely necessary

print("14. Select by column position:")
# Select first and third columns
df.select(df.columns[0], df.columns[2]).show()

# ==========================================================================
# PERFORMANCE CONSIDERATIONS AND BEST PRACTICES
# ==========================================================================

# 15. Column pruning for performance
# ==========================================================================
# PRODUCTION TIP: Select only the columns you need as early as possible
# This is an optimization technique called "column pruning" that can
# significantly improve performance, especially with wide tables
#
# GOOD: Select only needed columns early in your pipeline
needed_df = df.select("id", "name", "salary")
result1 = needed_df.filter(df.salary > 50000)

# BAD: Select all columns and use them later
# This forces Spark to process all columns through each operation
all_df = df.select("*")
result2 = all_df.filter(all_df.salary > 50000).select("id", "name", "salary")

# 16. Using select for validation
# ==========================================================================
# Select can be used to validate expected schema
# PRODUCTION TIP: This is a good practice before critical processing

print("16. Schema validation with select:")
try:
    # This will fail if the column doesn't exist, preventing invalid downstream processing
    validation_df = df.select("id", "missing_column")
    validation_df.show()
except Exception as e:
    print(f"Validation failed: {str(e)}")

# 17. Working with nested structures
# ==========================================================================
# Select is powerful for working with complex nested data structures
#If a column is a struct or map, you can select fields using dot notation in a string or using getField():
# Let's create a nested DataFrame to demonstrate:

# Create a DataFrame with a struct column
nested_df = df.select(
    "id",
    "name",
    F.struct("age", "salary", "department").alias("employee_info")
)

print("17. Working with nested data:")
nested_df.show(truncate=False)
nested_df.printSchema()

# Accessing nested fields
print("Accessing nested fields:")
nested_df.select(
    "id",
    "name",
    "employee_info.age",
    "employee_info.salary"
).show()

# Flattening nested structures
print("Flattening nested structures:")
nested_df.select(
    "id",
    "name",
    "employee_info.*"  # Extracts all fields from the struct
).show()

# ==========================================================================
# EDGE CASES AND TROUBLESHOOTING
# ==========================================================================

# 18. Handling special characters in column names
# ==========================================================================
# PRODUCTION PITFALL: Column names with spaces or special chars need special handling
# Create a DataFrame with problematic column names

problem_df = spark.createDataFrame(
    [(1, "John", 30), (2, "Alice", 25)],
    ["id", "employee name", "age.years"]  # Problematic column names
)

print("18. DataFrame with problematic column names:")
problem_df.show()

# Using backticks to escape special characters
print("Using backticks for special column names:")
problem_df.select(
    "`id`",
    "`employee name`",
    "`age.years`"
).show()

# Alternative: Use column objects
"""
print("Using column objects for special column names:")
problem_df.select(
    problem_df["id"],
    problem_df["employee name"],
    problem_df["age.years"]
).show()
"""


# 19. Case sensitivity issues
# ==========================================================================
# PITFALL: Column references are case-sensitive by default
# PRODUCTION TIP: Be consistent with casing throughout your code

print("19. Case sensitivity demonstration:")
try:
    # This will fail because 'NAME' doesn't match the actual column name 'name'
    df.select("id", "NAME").show()
except Exception as e:
    print(f"Case sensitivity error: {str(e)}")

    # Correct approach
    df.select("id", "name").show()

# 20. Select with duplicate column names
# ==========================================================================
# PITFALL: Selecting the same column multiple times creates duplicates
# Spark allows this but it can cause problems later

print("20. Duplicate column names:")
duplicate_df = df.select("id", "name", "name")
duplicate_df.show()
print("Schema with duplicates:")
duplicate_df.printSchema()

# PRODUCTION TIP: Always rename duplicates to avoid confusion
better_df = df.select("id", "name", df["name"].alias("name_copy"))
better_df.show()

# ==========================================================================
# COMBINING SELECT WITH OTHER OPERATIONS
# ==========================================================================

# 21. Chaining select with other transformations
# ==========================================================================
# Select can be part of a transformation chain
# PRODUCTION TIP: Place select operations strategically to optimize performance

print("21. Chaining operations:")
result = df.select("id", "name", "age", "salary") \
    .filter(df["age"].isNotNull()) \
    .orderBy("age") \
    .limit(3)
result.show()

# 22. Select after groupBy
# ==========================================================================
# Select is often used for cleaning up aggregation results
# PRODUCTION TIP: Rename columns to make aggregation results clearer

print("22. Select after groupBy:")
df.groupBy("department") \
    .agg(
    F.count("id").alias("employee_count"),
    F.avg("salary").alias("avg_salary"),
    F.min("age").alias("min_age"),
    F.max("age").alias("max_age")
) \
    .select(  # Clean up column names and format
    "department",
    "employee_count",
    F.round("avg_salary", 2).alias("average_salary"),
    F.coalesce(F.col("min_age"), F.lit("N/A")).alias("youngest_age"),
    F.coalesce(F.col("max_age"), F.lit("N/A")).alias("oldest_age")
) \
    .show()

# ==========================================================================
# CONCLUSION AND SUMMARY
# ==========================================================================
# Key takeaways about PySpark DataFrame select:
#
# 1. Syntax options:
#    - String column names: df.select("col1", "col2")
#    - Column objects: df.select(df["col1"], df["col2"])
#    - SQL expressions: df.selectExpr("col1", "col1 + col2 as sum")
#
# 2. Best practices:
#    - Select only needed columns early in your pipeline
#    - Use consistent syntax throughout your code
#    - Rename columns to clearly indicate transformations
#    - Handle NULL values appropriately
#
# 3. Common pitfalls:
#    - Case sensitivity in column names
#    - Special characters in column names need backticks
#    - Type conversion failures can create NULLs
#    - Position-based selection can break with schema changes
#
# 4. Performance tips:
#    - Early column pruning improves performance
#    - Chain operations effectively
#    - Prefer built-in functions over UDFs when possible
#
# 5. Advanced techniques:
#    - Working with nested structures
#    - Dynamic column selection
#    - Pattern-based column selection

# Stop the SparkSession
spark.stop()



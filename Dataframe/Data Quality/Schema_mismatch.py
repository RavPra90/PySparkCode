from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, lit

# Initialize Spark session
spark = SparkSession.builder.appName("SchemaMatchDemo").getOrCreate()

# Target Schema - what we want our data to look like
target_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("salary", DoubleType(), True)
])

# =============================================================================
# PROBLEM 1: Data Type Mismatches
# =============================================================================
print("PROBLEM 1: Data ype Mismatches")
print("Issue: Age is string, salary is int instead of expected types")

# Create dataset with wrong data types
data1 = [(1, "John", "john@email.com", "25", 50000),
         (2, "Mary", "mary@email.com", "30", 60000)]
df1_before = spark.createDataFrame(data1, ["id", "name", "email", "age", "salary"])

print("\nBEFORE:")
df1_before.printSchema()
df1_before.show()

# Fix data types
df1_after = df1_before.withColumn("age", col("age").cast(IntegerType())) \
                     .withColumn("salary", col("salary").cast(DoubleType()))

print("AFTER:")
df1_after.printSchema()
df1_after.show()

# =============================================================================
# PROBLEM 2: Missing Columns
# =============================================================================
print("\nPROBLEM 2: Missing Columns")
print("Issue: Email column is missing from the dataset")

# Create dataset missing email column
data2 = [(3, "Alice", 28, 55000.0),
         (4, "Bob", 32, 65000.0)]
df2_before = spark.createDataFrame(data2, ["id", "name", "age", "salary"])

print("\nBEFORE:")
df2_before.printSchema()
df2_before.show()

# Add missing email column with null values
df2_after = df2_before.withColumn("email", lit(None).cast(StringType())) \
                     .select("id", "name", "email", "age", "salary")

print("AFTER:")
df2_after.printSchema()
df2_after.show()

# =============================================================================
# PROBLEM 3: Extra Columns
# =============================================================================
print("\nPROBLEM 3: Extra Columns")
print("Issue: Dataset has extra 'phone' column that we don't need")

# Create dataset with extra column
data3 = [(5, "Charlie", "charlie@email.com", 35, 70000.0, "555-1234"),
         (6, "Diana", "diana@email.com", 29, 58000.0, "555-5678")]
df3_before = spark.createDataFrame(data3, ["id", "name", "email", "age", "salary", "phone"])

print("\nBEFORE:")
df3_before.printSchema()
df3_before.show()

# Remove extra column by selecting only needed columns
df3_after = df3_before.select("id", "name", "email", "age", "salary")

print("AFTER:")
df3_after.printSchema()
df3_after.show()

# =============================================================================
# PROBLEM 4: Column Name Mismatches (Case Sensitivity)
# =============================================================================
print("\nPROBLEM 4: Column Name Case Mismatches")
print("Issue: Column names are in uppercase instead of lowercase")

# Create dataset with wrong case column names
data4 = [(7, "Eve", "eve@email.com", 31, 62000.0),
         (8, "Frank", "frank@email.com", 27, 56000.0)]
df4_before = spark.createDataFrame(data4, ["ID", "NAME", "EMAIL", "AGE", "SALARY"])

print("\nBEFORE:")
df4_before.printSchema()
df4_before.show()

# Fix column names by renaming to lowercase
df4_after = df4_before.withColumnRenamed("ID", "id") \
                     .withColumnRenamed("NAME", "name") \
                     .withColumnRenamed("EMAIL", "email") \
                     .withColumnRenamed("AGE", "age") \
                     .withColumnRenamed("SALARY", "salary")

print("AFTER:")
df4_after.printSchema()
df4_after.show()

# =============================================================================
# PROBLEM 5: Column Reordering
# =============================================================================
print("\nPROBLEM 5: Column Reordering")
print("Issue: Columns are in wrong order")

# Create dataset with columns in wrong order
data5 = [("grace@email.com", 33, "Grace", 68000.0, 9),
         ("henry@email.com", 25, "Henry", 51000.0, 10)]
df5_before = spark.createDataFrame(data5, ["email", "age", "name", "salary", "id"])

print("\nBEFORE:")
df5_before.printSchema()
df5_before.show()

# Reorder columns to match target schema
df5_after = df5_before.select("id", "name", "email", "age", "salary")

print("AFTER:")
df5_after.printSchema()
df5_after.show()

# =============================================================================
# FINAL DEMONSTRATION: Union All Fixed DataFrames
# =============================================================================
print("\nFINAL RESULT: All schemas now match - can union together")

# Union all fixed dataframes
final_df = df1_after.union(df2_after).union(df3_after).union(df4_after).union(df5_after)

print("Combined DataFrame:")
final_df.printSchema()
final_df.show()

print("Total records:", final_df.count())

# Clean up
spark.stop()

"""
KEY SOLUTIONS SUMMARY:

1. Data Type Mismatches: Use cast() to convert types
   df.withColumn("col_name", col("col_name").cast(TargetType()))

2. Missing Columns: Add with lit() and null values
   df.withColumn("missing_col", lit(None).cast(TargetType()))

3. Extra Columns: Use select() to pick only needed columns
   df.select("col1", "col2", "col3")

4. Column Name Mismatches: Use withColumnRenamed()
   df.withColumnRenamed("OLD_NAME", "new_name")

5. Column Reordering: Use select() with correct order
   df.select("col1", "col2", "col3")

EDGE CASES TO CONSIDER:
- Null values during type conversion
- Invalid data that cannot be cast
- Very large datasets with performance implications
- Nested schemas with struct types
- Array and map type mismatches

PRODUCTION TIPS:
- Validate schemas before processing
- Log all transformations for debugging
- Use try-catch for error handling
- Test with sample data first
- Consider using Delta Lake for schema evolution
"""

"""
#PySpark #BigData #DataEngineering #ApacheSpark #SchemaEvolution 
#DataProcessing #ETL #DataPipeline #Python #DataIntegration 
#SparkSQL #DataQuality #DistributedComputing #TechTips #DataScience
"""
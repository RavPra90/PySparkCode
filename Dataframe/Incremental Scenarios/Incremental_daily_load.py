from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date, when, isnan, isnull, coalesce, lit, concat, date_format
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
import datetime

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("IncrementalDataLoad") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

#  Step 1: Create dummy datasets for demonstration

# Define schema for consistent data structure
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("transaction_id", StringType(), True),
    StructField("amount", IntegerType(), True),
    StructField("transaction_date", DateType(), True),
    StructField("status", StringType(), True)
])

# Existing data in target table (simulating historical data)
existing_data = [
    ("C001", "T001", 100, datetime.date(2024, 1, 1), "completed"),
    ("C002", "T002", 200, datetime.date(2024, 1, 1), "completed"),
    ("C003", None, 150, datetime.date(2024, 1, 2), "pending"),  # Null transaction_id
    ("C004", "T004", None, datetime.date(2024, 1, 2), "failed")  # Null amount
]

# New daily data to be loaded (contains duplicates and nulls)
new_daily_data = [
    ("C002", "T002", 200, datetime.date(2024, 1, 1), "completed"),  # Duplicate
    ("C005", "T005", 300, datetime.date(2024, 1, 3), "completed"),  # New record
    ("C006", None, 250, datetime.date(2024, 1, 3), "pending"),     # Null transaction_id
    ("C007", "T007", None, datetime.date(2024, 1, 3), "failed"),   # Null amount
    ("C005", "T005", 300, datetime.date(2024, 1, 3), "completed")  # Duplicate in new data
]

# Create DataFrames
target_df = spark.createDataFrame(existing_data, schema)
daily_df = spark.createDataFrame(new_daily_data, schema)

print("Target Table (Existing Data):")
target_df.show()
print("Daily Data (New Incoming Data):")
daily_df.show()

#  Step 2: Handle Null Values Before Processing

# Strategy: Replace nulls with meaningful defaults or filter them out
daily_df_cleaned = daily_df.withColumn(
    "transaction_id",
    # Replace null transaction_id with generated ID
    when(col("transaction_id").isNull(),
         concat(lit("GEN_"), col("customer_id"), lit("_"),
                date_format(col("transaction_date"), "yyyyMMdd")))
    .otherwise(col("transaction_id"))
).withColumn(
    "amount",
    # Replace null amounts with 0 or filter out based on business logic
    coalesce(col("amount"), lit(0))
).withColumn(
    "status",
    # Ensure status is never null
    coalesce(col("status"), lit("unknown"))
)

print("Daily Data After Null Handling:")
daily_df_cleaned.show()

# Step 3: Remove Duplicates Within New Data

# Remove duplicates within the daily data itself
# Keep the first occurrence of each unique combination
daily_df_deduped = daily_df_cleaned.dropDuplicates([
    "customer_id", "transaction_id", "transaction_date"
])

print("Daily Data After Internal Deduplication:")
daily_df_deduped.show()

#  Step 4: Identify and Filter Out Records Already in Target (Anti-Join to Exclude Existing Records)

# Create composite key for comparison
# This handles cases where business key is combination of multiple columns
target_with_key = target_df.withColumn(
    "composite_key",
    concat(col("customer_id"), lit("_"),
           col("transaction_id"), lit("_"),
           col("transaction_date"))
)

daily_with_key = daily_df_deduped.withColumn(
    "composite_key",
    concat(col("customer_id"), lit("_"),
           col("transaction_id"), lit("_"),
           col("transaction_date"))
)

# Anti-join: Keep only records from daily data that don't exist in target
# This is more efficient than using NOT IN or NOT EXISTS
new_records_only = daily_with_key.join(
    target_with_key.select("composite_key"),
    on="composite_key",
    how="left_anti"  # Left anti-join excludes matching records
).drop("composite_key")  # Remove helper column

print("New Records Only (After Anti-Join):")
new_records_only.show()

# Step 5: Final Incremental Load

# Union new records with existing data
final_df = target_df.union(new_records_only)

print("Final Dataset After Incremental Load:")
final_df.orderBy(col("customer_id")).show()

#  Step 6: Data Quality Checks and Metrics
print(f"Original target records: {target_df.count()}")
print(f"Daily data received: {daily_df.count()}")
print(f"After null handling: {daily_df_cleaned.count()}")
print(f"After deduplication: {daily_df_deduped.count()}")
print(f"New records to load: {new_records_only.count()}")
print(f"Final record count: {final_df.count()}")

# Clean up
spark.stop()

# 🎯 Key Cases Handled:
# 1. Null values in critical columns (transaction_id, amount)
# 2. Duplicates within incoming daily data
# 3. Records already existing in target table
# 4. Composite key matching for complex business keys
# 5. Data quality metrics and monitoring

# 💡 Step-by-Step Approach:
# 1. Data Preparation: Create schemas and load data
# 2. Null Handling: Clean and standardize null values
# 3. Internal Deduplication: Remove duplicates within new data
# 4. Anti-Join: Exclude records already in target
# 5. Union: Combine new records with existing data
# 6. Quality Checks: Monitor data pipeline metrics
# 7. Optimized Write: Partition and write efficiently

# 🔥 Developer Tips:
# 1. Use anti-join instead of NOT IN for better performance
# 2. Handle nulls early in the pipeline to avoid issues
# 3. Create composite keys for complex business logic
# 4. Monitor data quality metrics at each step
# 5. Use coalesce() for better write performance
# 6. Consider partitioning strategy for large datasets
# 7. Enable adaptive query execution for optimization
# 8. Use broadcast joins for small lookup tables
# 9. Cache intermediate results if used multiple times
# 10. Always validate data quality after incremental loads

#LinkedInHashtags: #PySpark #BigData #DataEngineering #ETL #ApacheSpark #DataPipeline #IncrementalLoad #DataQuality #Python #DataScience #Analytics #DataProcessing #SparkSQL #DataWarehouse #RealTimeData
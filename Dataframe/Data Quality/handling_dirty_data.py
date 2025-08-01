from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from datetime import datetime, timedelta

# Initialize Spark session
spark = SparkSession.builder.appName("OutOfRangeValueHandling").getOrCreate()

# Create a schema for our dummy dataset
# Schema helps define structure and expected data types upfront
schema = StructType([
    StructField("user_id", StringType(), False),  # False = not nullable
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("signup_date", TimestampType(), True)
])

# Current time for reference (used for future date comparison)
current_time = datetime.now()

# Create dummy data with various edge cases
data = [
    ("U001", "Alex", 25, current_time - timedelta(days=30)),     # Valid data
    ("U002", "Blake", -5, current_time - timedelta(days=10)),    # Negative age!
    ("U003", "Casey", 150, current_time - timedelta(days=5)),    # Unrealistically high age
    ("U004", "Dana", None, current_time - timedelta(days=2)),    # Missing age
    ("U005", "Eli", 22, current_time + timedelta(days=5)),       # Future signup date!
    ("U006", "Fran", 32, None)                                   # Missing date
]

# Create DataFrame with our schema and data
df = spark.createDataFrame(data, schema)

# Display original data
print("Original DataFrame:")
df.show(truncate=False)

# Handle out-of-range values using multiple techniques
cleaned_df = df.withColumn(
    "age_cleaned",
    # Handle age outliers with WHEN-OTHERWISE (similar to if-else)
    F.when(
        (F.col("age").isNull()) | (F.col("age") < 0) | (F.col("age") > 100),
        None  # Replace invalid ages with NULL
    ).otherwise(F.col("age"))
).withColumn(
    "age_with_default",
    # Provide a default value (30) when age is invalid
    F.when(
        (F.col("age").isNull()) | (F.col("age") < 0) | (F.col("age") > 100),
        30
    ).otherwise(F.col("age"))
).withColumn(
    "signup_date_cleaned",
    # Fix future dates by setting them to current time
    F.when(
        F.col("signup_date") > F.lit(current_time),
        F.lit(current_time)
    ).otherwise(F.col("signup_date"))
).withColumn(
    # Add flag to identify records that had invalid values
    "has_invalid_data",
    (F.col("age").isNull()) |
    (F.col("age") < 0) |
    (F.col("age") > 100) |
    (F.col("signup_date") > F.lit(current_time)) |
    (F.col("signup_date").isNull())
)

# Display cleaned data
print("\nCleaned DataFrame:")
cleaned_df.select("user_id", "name", "age", "age_cleaned", "age_with_default",
                 "signup_date", "signup_date_cleaned", "has_invalid_data").show(truncate=False)

# Example of filtering out records with invalid data (if needed)
valid_records = cleaned_df.filter(~F.col("has_invalid_data"))
print("\nOnly Valid Records:")
valid_records.select("user_id", "name", "age", "signup_date").show(truncate=False)


# Save the cleaned DataFrame (commented out for demo)
# cleaned_df.write.mode("overwrite").parquet("path/to/cleaned_data")
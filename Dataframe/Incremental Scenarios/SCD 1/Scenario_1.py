from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_date
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, TimestampType
import datetime

# Initialize Spark session (no external dependencies)
spark = SparkSession.builder \
    .appName("SCD_Type1_InMemory_Operations") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Step 1: Define schema for customer profiles (strongly typed approach)
customer_schema = StructType([
    StructField("customer_id", IntegerType(), False),      # Primary key - not nullable
    StructField("email", StringType(), False),            # Email - not nullable
    StructField("first_name", StringType(), False),       # First name - not nullable
    StructField("last_name", StringType(), False),        # Last name - not nullable
    StructField("signup_date", DateType(), False),        # Signup date - not nullable
    StructField("last_updated", TimestampType(), True)    # Last updated - nullable for initial load
])


# Step 2: Create existing customer data with proper schema
existing_customers_data = [
    (1001, "john.doe@email.com", "John", "Doe",datetime.date(2024, 1, 15)
     ,datetime.datetime(2024, 1, 15, 10, 0, 0)  # datetime object
    ),
    (1002, "jane.smith@email.com", "Jane", "Smith",datetime.date(2024, 2, 20),
     datetime.datetime(2024, 2, 20, 11, 30, 0)
    ),
    (1005, "charlie.davis@email.com", "Charlie", "Davis",datetime.date(2024, 1, 1),
     datetime.datetime(2024, 1, 1, 9, 15, 0)
    )
]
# Create existing customer DataFrame with schema enforcement
existing_df = spark.createDataFrame(existing_customers_data, schema=customer_schema)

print("Existing Customer Table:")
existing_df.show(truncate=False)

# Step 3: Create daily JSON feed data (simulating incoming updates)
daily_updates_data = [
    (1001, "john.doe.updated@email.com", "John", "Doe", "2024-01-15"),           # Email update
    (1002, "jane.smith@email.com", "Jane", "Smith-Johnson", "2024-02-20"),       # Last name change
    (1003, "bob.wilson@email.com", "Robert", "Wilson", "2024-03-10"),            # New customer
    (1004, "alice.brown@email.com", "Alice", "Brown", "2024-04-05")             # New customer
]

# Create daily feed DataFrame with partial schema (no last_updated yet)
daily_feed_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("email", StringType(), False),
    StructField("first_name", StringType(), False),
    StructField("last_name", StringType(), False),
    StructField("signup_date", StringType(), False)  # String initially, will convert to date
])

daily_feed_df = spark.createDataFrame(daily_updates_data, schema=daily_feed_schema)

# Step 4: Transform daily feed to match target schema
# Convert signup_date from string to date and add last_updated timestamp
incoming_df = daily_feed_df \
    .withColumn("signup_date", to_date(col("signup_date"), "yyyy-MM-dd")) \
    .withColumn("last_updated", current_timestamp())

print("Daily Feed Data (After Schema Transformation):")
incoming_df.show(truncate=False)

# Step 5: Cache existing table for multiple join operations
# This is critical for performance when performing multiple operations
existing_df.cache()
print("Existing table cached for performance optimization")

# Step 6: SCD Type 1 Logic - Identify customers that need updates
# Inner join to find existing customers that appear in daily feed
customers_to_update = existing_df.alias("existing").join(
    incoming_df.alias("incoming"),
    col("existing.customer_id") == col("incoming.customer_id"),
    "inner"
).select(
    # Select all columns from incoming data (SCD Type 1 - overwrite with new values)
    col("incoming.customer_id"),
    col("incoming.email"),
    col("incoming.first_name"),
    col("incoming.last_name"),
    col("existing.signup_date"),        # Keep original signup_date (immutable)
    col("incoming.last_updated")        # Update timestamp from processing
)

print("Customers Being Updated (SCD Type 1 Overwrite):")
customers_to_update.show(truncate=False)

# Step 7: Find customers that remain unchanged
# Left anti-join to get existing customers NOT in the daily feed
customers_unchanged = existing_df.join(
    incoming_df,
    existing_df.customer_id == incoming_df.customer_id,
    "left_anti"
)

print("Customers Remaining Unchanged:")
customers_unchanged.show(truncate=False)

# Step 8: Find completely new customers (inserts)
# Left anti-join to get customers in daily feed but NOT in existing table
new_customers = incoming_df.join(
    existing_df,
    incoming_df.customer_id == existing_df.customer_id,
    "left_anti"
)

print("New Customers Being Inserted:")
new_customers.show(truncate=False)

# Step 9: Create final SCD Type 1 result using UNION
# Combine: Updated customers + Unchanged customers + New customers
final_customer_table = customers_to_update \
    .union(customers_unchanged) \
    .union(new_customers)

print("Final Customer Table After SCD Type 1 Processing:")
final_customer_table.orderBy("customer_id").show(truncate=False)

# Step 10: Show processing statistics
print("Processing Statistics:")
print(f"Records Updated: {customers_to_update.count()}")
print(f"Records Unchanged: {customers_unchanged.count()}")
print(f"Records Inserted: {new_customers.count()}")
print(f"Total Records: {final_customer_table.count()}")

# Step 11: Show only records updated today (performance query example)
print("Records Updated Today:")
final_customer_table.filter(
    col("last_updated").cast("date") == current_timestamp().cast("date")
).show(truncate=False)

# Step 12: Clean up cached resources
existing_df.unpersist()
print("Cache cleaned up")

spark.stop()
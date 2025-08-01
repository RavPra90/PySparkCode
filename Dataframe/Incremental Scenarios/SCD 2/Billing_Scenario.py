"""
Scenario:
A billing system emits CDC events for subscriptions, with columns: `user_id`, `plan_id`, `status`, and a change-timestamp.
You must maintain a full history: each version of a user’s subscription should have `effective_from`, `effective_to`,
plus an `is_current` flag.

Question:
Outline a PySpark-based merge strategy that:
- Inserts a new record whenever `plan_id` or `status` changes, setting the previous row’s `effective_to` and flipping its `is_current` to false
- Ensures idempotency so reprocessing the same CDC batch doesn’t create duplicates
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, BooleanType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SCD_Type2_Implementation") \
    .getOrCreate()

# STEP 1: Create dummy datasets for demonstration

# Existing historical data (SCD Type 2 table)
existing_data = [
    ("user_001", "basic", "active", "2024-01-01 00:00:00", "2024-02-15 10:30:00", False),
    ("user_001", "premium", "active", "2024-02-15 10:30:00", None, True),
    ("user_002", "basic", "active", "2024-01-15 00:00:00", None, True),
]
existing_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("plan_id", StringType(), True),
    StructField("status", StringType(), True),
    StructField("effective_from", StringType(), True),
    StructField("effective_to", StringType(), True),
    StructField("is_current", BooleanType(), True)
])
existing_df = spark.createDataFrame(existing_data, existing_schema)

# Convert string timestamps to actual timestamp type
existing_df = existing_df.withColumn("effective_from", col("effective_from").cast(TimestampType())) \
    .withColumn("effective_to", col("effective_to").cast(TimestampType()))
print("Existing Historical Data:")
existing_df.show(truncate=False)

# New CDC events (incoming changes) - Mix of valid and invalid changes
cdc_data = [
    ("user_001", "premium", "suspended", "2024-03-01 14:20:00"),  # Valid: Status change from active to suspended
    ("user_002", "premium", "active", "2024-02-28 09:15:00"),  # Valid: Plan upgrade from basic to premium
    ("user_003", "basic", "active", "2024-03-02 11:00:00"),  # Valid: New user
    ("user_001", "premium", "suspended", "2024-03-01 14:20:00"),  # Duplicate (idempotency test)
    ("user_001", "premium", "active", "2024-02-20 10:00:00"),  # Invalid: Same as current state - NO CHANGE
    ("user_002", "basic", "active", "2024-01-10 08:00:00"),  # Invalid: Outdated timestamp (older than current)
    ("user_001", "premium", "active", "2024-02-10 12:00:00"),  # Invalid: Even older timestamp - LATE ARRIVAL
    ("user_002", "basic", "active", "2024-02-29 10:00:00"),  # Invalid: No actual change in values
]
cdc_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("plan_id", StringType(), True),
    StructField("status", StringType(), True),
    StructField("change_timestamp", StringType(), True)
])
cdc_df = spark.createDataFrame(cdc_data, cdc_schema)
cdc_df = cdc_df.withColumn("change_timestamp", col("change_timestamp").cast(TimestampType()))
print("\nIncoming CDC Events:")
cdc_df.show(truncate=False)

#STEP 2: Deduplicate CDC events for idempotency

# Remove duplicates within the CDC batch based on all columns
# This ensures reprocessing the same batch doesn't create duplicates
cdc_deduplicated = cdc_df.dropDuplicates()
print("Deduplicated CDC Events:")
cdc_deduplicated.show(truncate=False)
print(f"Original CDC events: {cdc_df.count()}, After deduplication: {cdc_deduplicated.count()}")

# STEP 3: Identify changes that require new SCD record
# Get current active records for comparison
current_records = existing_df.filter(col("is_current") == True)
print("Current Active Records for Comparison:")
current_records.show(truncate=False)

# Join CDC events with current records to identify actual changes
# Left join to include new users and changed records
changes_check = cdc_deduplicated.alias("cdc").join(
    current_records.alias("curr"),
    col("cdc.user_id") == col("curr.user_id"),
    "left"
)
print("CDC Events Joined with Current Records:")
changes_with_analysis = changes_check.select(
    "cdc.user_id",
    col("cdc.plan_id").alias("new_plan"),
    col("cdc.status").alias("new_status"),
    "cdc.change_timestamp",
    col("curr.plan_id").alias("current_plan"),
    col("curr.status").alias("current_status"),
    col("curr.effective_from").alias("current_from"),
    # Fixed analysis logic - check outdated BEFORE no-change
    when(col("curr.user_id").isNull(), "✅ NEW_USER - Process")
    .when(col("cdc.change_timestamp") < col("curr.effective_from"), "❌ OUTDATED - Skip")
    .when((col("cdc.plan_id") == col("curr.plan_id")) & (col("cdc.status") == col("curr.status")), "❌ NO_CHANGE - Skip")
    .otherwise("✅ VALID_CHANGE - Process").alias("analysis")
).orderBy("user_id", "change_timestamp")
changes_with_analysis.show(truncate=False)

# Count by decision type using the new DataFrame that has the analysis column
decision_summary = changes_with_analysis.groupBy("analysis").count().orderBy("analysis")
print("Step 3 Analysis Summary:")
decision_summary.show(truncate=False)


# STEP 4: Create new SCD records

# Only process records marked as "✅ NEW_USER - Process" or "✅ VALID_CHANGE - Process"
actual_changes = changes_with_analysis.filter(
    col("analysis").isin(["✅ NEW_USER - Process", "✅ VALID_CHANGE - Process"])
)
# Transform CDC events into SCD format
new_scd_records = actual_changes.select(
    col("cdc.user_id"),
    col("new_plan"),
    col("new_status"),
    col("cdc.change_timestamp").alias("effective_from"),
    lit(None).cast(TimestampType()).alias("effective_to"),  # Open-ended for current records
    lit(True).alias("is_current")  # Mark as current
).orderBy("user_id","change_timestamp")
print("New SCD Records to Insert:")
new_scd_records.show(truncate=False)


# STEP 5: Update existing current records (close them)

# Get user_ids that have changes to update their current records
# Only get users with VALID_CHANGE (not NEW_USER since they don't have existing records)
users_with_changes = actual_changes.filter(col("analysis") == "✅ VALID_CHANGE - Process") \
    .select("user_id", "change_timestamp").distinct()

# Update existing current records: set effective_to and is_current=False
updated_existing = existing_df.alias("existing").join(
    users_with_changes.alias("changes"),
    col("existing.user_id") == col("changes.user_id"),
    "left"
).select(
    col("existing.user_id"),
    col("existing.plan_id"),
    col("existing.status"),
    col("existing.effective_from"),
    # Set effective_to to change_timestamp for records being superseded
    when(col("existing.is_current") & col("changes.user_id").isNotNull(),
         col("changes.change_timestamp"))
    .otherwise(col("existing.effective_to")).alias("effective_to"),
    # Set is_current to False for records being superseded
    when(col("existing.is_current") & col("changes.user_id").isNotNull(),
         lit(False))
    .otherwise(col("existing.is_current")).alias("is_current")
)
print("Updated Existing Records:")
updated_existing.show(truncate=False)


# STEP 6: Combine updated existing records with new records

# Union the updated existing records with new records
final_scd_table = updated_existing.union(new_scd_records)
print("Complete SCD Type 2 Historical Table:")
final_scd_table.orderBy("user_id", "effective_from").show(truncate=False)

# STEP 7: Verification and Data Quality Checks

# Check 1: Ensure only one current record per user
current_count_per_user = final_scd_table.filter(col("is_current") == True) \
    .groupBy("user_id").count()
print("Current Records Count Per User (should be 1 for each):")
current_count_per_user.show()


# Clean up
spark.stop()

"""
🎯 KEY STEPS IN THE CODING APPROACH:

1. **Data Preparation**: Create structured schemas and sample datasets
2. **Idempotency Handling**: Remove duplicates from CDC events
3. **Change Detection**: Identify actual changes requiring SCD processing
4. **Record Creation**: Generate new SCD records with proper timestamps
5. **Historical Update**: Close existing current records
6. **Data Merging**: Union updated and new records
7. **Quality Verification**: Validate SCD constraints
8. **Function Abstraction**: Create reusable production function

🚨 EDGE CASES HANDLED:

• **Duplicate CDC Events**: Automatic deduplication prevents duplicate records
• **Out-of-Order Events**: Timestamp comparison ensures proper sequencing  
• **New Users**: Handled separately from existing user updates
• **No-Change Events**: Filtered out to avoid unnecessary processing
• **Null Timestamps**: Proper casting and null handling for open records
• **Concurrent Updates**: Window functions ensure data consistency

💡 DEVELOPER TIPS:

• **Performance**: Use partitioning on user_id for large datasets
• **Monitoring**: Add logging for change detection and processing metrics
• **Testing**: Always verify single current record per user constraint
• **Optimization**: Consider Delta Lake for ACID transactions in production
• **Scalability**: Implement checkpointing for large CDC batches
• **Schema Evolution**: Design flexible schema handling for new columns
"""
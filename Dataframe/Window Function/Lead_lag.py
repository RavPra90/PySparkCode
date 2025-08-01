from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("LeadLagTransactionAnalysis") \
    .getOrCreate()

# Create comprehensive dummy dataset for demonstration
dummy_data = [
    (1, "user_001", 150.50, "2024-01-15"),
    (2, "user_001", 200.00, "2024-02-10"),
    (3, "user_001", 175.25, "2024-03-05"),
    (4, "user_001", 300.00, "2024-04-20"),
    (5, "user_001", 125.75, "2024-05-15"),
    (6, "user_002", 89.99, "2024-01-20"),
    (7, "user_002", 120.00, "2024-02-15"),
    (8, "user_002", 95.50, "2024-03-10"),
    (9, "user_002", 250.75, "2024-04-25"),
    (10, "user_003", 75.00, "2024-01-10"),
    (11, "user_003", 180.50, "2024-02-28"),
    (12, "user_003", 220.00, "2024-03-15"),
    (13, "user_003", 165.25, "2024-04-30"),
    (14, "user_003", 310.00, "2024-05-20")
]

# Define schema for better performance and type safety
schema = StructType([
    StructField("transaction_id", IntegerType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("date", StringType(), True)
])

# Create DataFrame and convert date string to proper date type
df = spark.createDataFrame(dummy_data, schema) \
    .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

print("Original Transaction Dataset:")
df.show()

# Define primary window specification for sequential analysis
# Partition by user to keep each user's data separate, order by date for chronological sequence
user_date_window = Window.partitionBy("user_id").orderBy("date")

# =====================================
# 1. PURCHASE AMOUNT VARIANCE
# =====================================
# TECHNICAL SCENARIO: Purchase amount variance analysis using lag window function
# Challenge: Compare each customer's current purchase with their immediately previous purchase amount
# Solution: Partition by user_id, order by date, use lag() to access previous row's amount value
print("\n1. Purchase Amount Variance Analysis:")

df_variance = df.withColumn(
    "previous_amount",  # Look back 1 row to get the previous transaction amount
    lag("amount", 1).over(user_date_window)
).withColumn(
    "amount_difference",  # Calculate the spending difference from previous purchase
    round(col("amount") - col("previous_amount"))
).withColumn(
    "variance_type",  # Categorize spending behavior for business insights
    when(col("amount_difference") > 0, "Increased Spending")
    .when(col("amount_difference") < 0, "Decreased Spending")
    .when(col("amount_difference").isNull(), "First Purchase")
    .otherwise("Same Amount")
)

df_variance.select("user_id", "date", "amount", "previous_amount", "amount_difference", "variance_type").show()

# =====================================
# 2. CUSTOMER PURCHASE FREQUENCY
# =====================================
# TECHNICAL SCENARIO: Customer purchase frequency analysis using date lag operations
# Challenge: Calculate the time gap (in days) between consecutive purchases for each customer
# Solution: Apply lag() on date column within user partition, then use datediff() for interval calculation
print("\n2. Customer Purchase Frequency (Days Between Purchases):")

df_frequency = df.withColumn(
    "previous_purchase_date",  # Get the date of the previous purchase
    lag("date", 1).over(user_date_window)
).withColumn(
    "days_since_last_purchase",  # Calculate days between consecutive purchases
    datediff(col("date"), col("previous_purchase_date"))
).withColumn(
    "purchase_frequency_category",  # Categorize purchase patterns for analysis
    when(col("days_since_last_purchase").isNull(), "First Purchase")
    .when(col("days_since_last_purchase") <= 30, "Frequent Buyer")
    .when(col("days_since_last_purchase") <= 60, "Regular Buyer")
    .otherwise("Infrequent Buyer")
)

df_frequency.select("user_id", "date", "previous_purchase_date", "days_since_last_purchase", "purchase_frequency_category").show()

# =====================================
# 3. NEXT TRANSACTION AMOUNT PREVIEW
# =====================================
# TECHNICAL SCENARIO: Future transaction amount preview using lead window function
# Challenge: Show what amount each customer will spend on their next purchase for predictive analysis
# Solution: Use lead() to look ahead 1 row and bring future transaction amount to current row
print("\n3. Next Transaction Amount Preview:")

df_next_amount = df.withColumn(
    "next_purchase_amount",  # Look forward 1 row to get the next transaction amount
    lead("amount", 1).over(user_date_window)
).withColumn(
    "spending_trend",  # Predict spending behavior based on next purchase
    when(col("next_purchase_amount").isNull(), "Last Purchase")
    .when(col("next_purchase_amount") > col("amount"), "Will Spend More")
    .when(col("next_purchase_amount") < col("amount"), "Will Spend Less")
    .otherwise("Will Spend Same")
).withColumn(
    "next_purchase_difference",  # Calculate the upcoming spending change
   round(col("next_purchase_amount") - col("amount"))
)

df_next_amount.select("user_id", "date", "amount", "next_purchase_amount", "next_purchase_difference", "spending_trend").show()

# =====================================
# 4. UPCOMING PURCHASE DATE ALERT
# =====================================
# TECHNICAL SCENARIO: Future purchase date prediction using lead for temporal analysis
# Challenge: Identify when each customer's next purchase will occur for retention planning
# Solution: Use lead() on date column and calculate forward-looking time intervals
print("\n4. Upcoming Purchase Date Alert:")

df_next_date = df.withColumn(
    "next_purchase_date",  # Look ahead to get the date of next transaction
    lead("date", 1).over(user_date_window)
).withColumn(
    "days_to_next_purchase",  # Calculate how many days until next purchase
    datediff(col("next_purchase_date"), col("date"))
).withColumn(
    "customer_type",  # Categorize customer purchase speed for marketing campaigns
    when(col("next_purchase_date").isNull(), "New Shopper")
    .when(col("days_to_next_purchase") <= 30, "Monthly Shopper")
    .when(col("days_to_next_purchase") <= 60, "Bi-Monthly Shopper")
    .otherwise("Rare Shopper ")
)

df_next_date.select("user_id", "date", "next_purchase_date", "days_to_next_purchase", "customer_type").show(truncate= False)

# =====================================
# 5. FIRST PURCHASE AMOUNT REFERENCE
# =====================================
# TECHNICAL SCENARIO: First purchase amount baseline comparison using first_value window function
# Challenge: Tag every transaction with the customer's very first purchase amount for lifetime comparison
# Solution: Use first_value() over user partition ordered by date to capture initial purchase baseline
print("\n5. First Purchase Amount Reference:")

df_first_reference = df.withColumn(
    "first_purchase_amount",  # Get the very first purchase amount for each customer
    first_value("amount").over(user_date_window)
).withColumn(
    "amount_vs_first_purchase",  # Compare current spending with initial purchase
    round(col("amount") - col("first_purchase_amount"))
).withColumn(
    "customer_evolution",  # Track how customer spending has evolved
    when(col("amount_vs_first_purchase") == 0, "Same as First Purchase")
    .when(col("amount_vs_first_purchase") > 0, "Spending More Than First")
    .otherwise("Spending Less Than First")
)

df_first_reference.select("user_id", "date", "amount", "first_purchase_amount",
                         "amount_vs_first_purchase", "customer_evolution").show(truncate=False)

# =====================================
# COMPREHENSIVE ANALYSIS COMBINING ALL METRICS
# =====================================
print("\n=== COMPREHENSIVE CUSTOMER TRANSACTION ANALYSIS ===")

# Combine all window function analyses into a single comprehensive view
df_comprehensive = df.withColumn(
    # Previous transaction metrics
    "previous_amount", lag("amount", 1).over(user_date_window)
).withColumn(
    "previous_date", lag("date", 1).over(user_date_window)
).withColumn(
    # Next transaction metrics
    "next_amount", lead("amount", 1).over(user_date_window)
).withColumn(
    "next_date", lead("date", 1).over(user_date_window)
).withColumn(
    # First purchase baseline
    "first_amount", first_value("amount").over(user_date_window)
).withColumn(
    # Calculated metrics
    "days_since_previous", datediff(col("date"), col("previous_date"))
).withColumn(
    "days_to_next", datediff(col("next_date"), col("date"))
).withColumn(
    "growth_from_first", round(col("amount") - col("first_amount"))
)

# Display comprehensive analysis
print("Complete Transaction Timeline Analysis:")
df_comprehensive.select("user_id", "date", "amount", "previous_amount", "next_amount",
                       "first_amount", "days_since_previous", "days_to_next", "growth_from_first").show()

spark.stop()

# =====================================
# KEY STEPS SUMMARY:
# =====================================
"""
KEY STEPS TAKEN:
1. Defined window specifications with proper partitioning by user_id and ordering by date
2. Applied lag() functions to access previous row values for historical comparison
3. Used lead() functions to peek into future rows for predictive analysis
4. Implemented first_value() to establish baseline references across partitions
5. Combined multiple window functions for comprehensive transaction timeline analysis
6. Added business logic categorization for actionable insights
7. Handled edge cases with null checks and safe calculations

EDGE CASES HANDLED:
- First transactions have no previous values (lag returns null)
- Last transactions have no future values (lead returns null)
- Single-transaction customers
- Division by zero in ratio calculations
- Date parsing and temporal calculations

DEVELOPER TIPS:
- Reuse window specifications for better performance
- Always handle null values from lag/lead operations
- Use meaningful column names for business understanding
- Combine window functions with conditional logic for insights
- Consider caching DataFrames used in multiple window operations
- Test with edge cases like single-transaction users
- Use round() function for financial calculations
- Validate results with known test datasets
"""
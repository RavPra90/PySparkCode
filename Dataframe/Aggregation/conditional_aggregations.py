from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum , count, when, avg, month, year, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from datetime import date

# Initialize Spark session
spark = SparkSession.builder.appName("EcommerceConditionalAggregation").getOrCreate()

# Define schema for e-commerce dataset
schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_date", DateType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("order_status", StringType(), True),
    StructField("payment_method", StringType(), True)
])

# Create dummy dataset with realistic e-commerce data
data = [
    ("ORD001", "CUST001", date(2024, 1, 15), 2, 150.0, "Completed", "Credit Card"),
    ("ORD002", "CUST002", date(2024, 1, 20), 1, 250.0, "Pending", "PayPal"),
    ("ORD003", "CUST001", date(2024, 2, 10), 3, 80.0, "Returned", "Credit Card"),
    ("ORD004", "CUST003", date(2024, 2, 25), 1, 300.0, "Completed", "Bank Transfer"),
    ("ORD005", "CUST002", date(2024, 3, 5), 2, 120.0, "Cancelled", "PayPal"),
    ("ORD006", "CUST004", date(2024, 3, 12), 4, 75.0, "Completed", "Credit Card"),
    ("ORD007", "CUST003", date(2024, 3, 18), 1, 180.0, "Pending", "Bank Transfer"),
    ("ORD008", "CUST001", date(2024, 4, 8), 2, 90.0, "Returned", "PayPal"),
    ("ORD009", "CUST005", date(2024, 4, 22), 1, 350.0, "Completed", "Credit Card"),
    ("ORD010", "CUST004", date(2024, 5, 1), 3, 110.0, "Completed", "Bank Transfer")
]

# Create DataFrame from dummy data
df = spark.createDataFrame(data, schema)

# Add calculated column for total order value (quantity * unit_price)
df = df.withColumn("order_value", col("quantity") * col("unit_price"))

print("Sample Data:")
df.show()

# =============================================================================
# Q1: Total orders and revenue by payment method (Completed/Pending only)
# =============================================================================
print("\n=== Q1: Orders & Revenue by Payment Method (Completed/Pending) ===")

q1_result = (df
    # Filter for only Completed or Pending orders using conditional logic
    .filter(col("order_status").isin(["Completed", "Pending"]))
    # Group by payment method for aggregation
    .groupBy("payment_method")
    # Aggregate: count total orders and sum revenue (order_value)
    .agg(
        count("order_id").alias("total_orders"),
        sum("order_value").alias("total_revenue")
    )
    .orderBy("payment_method")
)
q1_result.show()

# =============================================================================
# Q2: Monthly revenue comparison between Credit Card and PayPal
# =============================================================================
print("\n=== Q2: Monthly Revenue - Credit Card vs PayPal ===")

q2_result = (df
    # Filter for only Credit Card and PayPal payments
    .filter(col("payment_method").isin(["Credit Card", "PayPal"]))
    # Add month-year column for grouping
    .withColumn("month_year", month(col("order_date")))
    # Group by month and payment method
    .groupBy("month_year", "payment_method")
    # Sum revenue for each group
    .agg(sum("order_value").alias("monthly_revenue"))
    # Pivot to show Credit Card and PayPal as separate columns
    .groupBy("month_year")
    .pivot("payment_method", ["Credit Card", "PayPal"])
    .agg(sum("monthly_revenue"))
    .orderBy("month_year")
)
q2_result.show()

# =============================================================================
# Q3: Percentage of returned orders by Customer
# =============================================================================
print("\n=== Q3: Percentage of Returned Orders by Customers ===")

q3_result = (df
    .groupBy("customer_id")
    .agg(
        # Count total orders for each payment method
        count("*").alias("total_orders"),
        # Count only returned orders using conditional aggregation
        sum(when(col("order_status") == "Returned", 1).otherwise(0)).alias("returned_orders")
    )
    # Calculate percentage: (returned_orders / total_orders) * 100
    .withColumn("return_percentage",
                spark_round((col("returned_orders") / col("total_orders")) * 100, 2))
    .orderBy("customer_id")
)
q3_result.show()

# =============================================================================
# Q4: Average order value comparison by customer (High vs Low value orders)
# =============================================================================
print("\n=== Q4: Customer Analysis - High vs Low Value Orders ===")

q4_result = (df
    .groupBy("customer_id")
    .agg(
        # Calculate average for high-value orders (>$200) using conditional aggregation
        avg(when(col("order_value") > 200, col("order_value"))).alias("avg_high_value"),
        # Calculate average for low-value orders (≤$200)
        avg(when(col("order_value") <= 200, col("order_value"))).alias("avg_low_value"),
        # Count high-value orders
        sum(when(col("order_value") > 200, 1).otherwise(0)).alias("high_value_count"),
        # Count low-value orders
        sum(when(col("order_value") <= 200, 1).otherwise(0)).alias("low_value_count")
    )
    # Round averages to 2 decimal places for better readability
    .withColumn("avg_high_value", spark_round(col("avg_high_value"), 2))
    .withColumn("avg_low_value", spark_round(col("avg_low_value"), 2))
    .orderBy("customer_id")
)
q4_result.show()

# =============================================================================
# Q5: Monthly pivot with payment method percentages
# =============================================================================
print("\n=== Q5: Monthly Payment Method Distribution with Percentages ===")

# First, create monthly pivot of order counts
monthly_pivot = (df
    .withColumn("month_year", month(col("order_date")))
    .groupBy("month_year")
    .pivot("payment_method", ["Credit Card", "PayPal", "Bank Transfer"])
    .agg(count("*"))
    .fillna(0)  # Fill null values with 0 for months with no orders
)

# Calculate total orders per month and add percentage columns
q5_result = (monthly_pivot
    # Calculate total orders per month by summing all payment methods
    .withColumn("total_monthly_orders",
                col("Credit Card") + col("PayPal") + col("Bank Transfer"))
    # Calculate percentage for each payment method
    .withColumn("Credit_Card_Pct",
                spark_round((col("Credit Card") / col("total_monthly_orders")) * 100, 1))
    .withColumn("PayPal_Pct",
                spark_round((col("PayPal") / col("total_monthly_orders")) * 100, 1))
    .withColumn("Bank_Transfer_Pct",
                spark_round((col("Bank Transfer") / col("total_monthly_orders")) * 100, 1))
    .orderBy("month_year")
)
q5_result.show()

# Clean up resources
spark.stop()

# =============================================================================
# KEY STEPS IN THE CODING APPROACH:
# =============================================================================
# 1. Data Preparation: Create schema and dummy dataset with realistic e-commerce data
# 2. Conditional Filtering: Use .filter() and .isin() for status-based filtering
# 3. Conditional Aggregation: Leverage when().otherwise() for conditional calculations
# 4. Pivot Operations: Use .pivot() to transform rows into columns for comparison
# 5. Percentage Calculations: Combine aggregations with mathematical operations
# 6. Data Enrichment: Add calculated columns like order_value and month_year
# 7. Result Formatting: Use round() for better readability of decimal values

# =============================================================================
# EDGE CASES TO CONSIDER:
# =============================================================================
# 1. Null Values: Use .fillNa() when pivoting to handle missing combinations
# 2. Division by Zero: Check for zero denominators when calculating percentages
# 3. Empty Groups: Some payment methods might have no orders in certain periods
# 4. Data Types: Ensure proper casting when performing mathematical operations
# 5. Date Handling: Consider timezone issues and date format consistency

# =============================================================================
# DEVELOPER TIPS:
# =============================================================================
# 1. Always use col() function instead of string column names for better performance
# 2. Chain operations efficiently to minimize data shuffling
# 3. Use .cache() for DataFrames used multiple times in complex queries
# 4. Leverage Spark's lazy evaluation - transformations are not executed until action
# 5. Use .explain() to understand query execution plans for optimization
# 6. Consider partitioning by date columns for better performance with time-series data
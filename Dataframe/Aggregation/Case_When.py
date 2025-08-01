from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, count, avg, quarter, year
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, size
from datetime import date

# Initialize Spark session
spark = SparkSession.builder.appName("EcommerceAnalytics").getOrCreate()

# Define schema for better performance and type safety
schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_date", DateType(), True),
    StructField("product_category", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("order_status", StringType(), True),
    StructField("payment_method", StringType(), True)
])

# Create dummy dataset with realistic e-commerce data
data = [
    ("ORD001", "CUST001", date(2024, 1, 15), "Electronics", 2, 250.0, "Completed", "Credit Card"),
    ("ORD002", "CUST002", date(2024, 1, 20), "Clothing", 1, 75.0, "Completed", "PayPal"),
    ("ORD003", "CUST001", date(2024, 2, 10), "Books", 3, 180.0, "Cancelled", "Bank Transfer"),
    ("ORD004", "CUST001", date(2024, 2, 25), "Electronics", 1, 320.0, "Completed", "Credit Card"),
    ("ORD005", "CUST004", date(2024, 3, 5), "Home & Garden", 2, 45.0, "Pending", "PayPal"),
    ("ORD006", "CUST002", date(2024, 3, 12), "Clothing", 4, 150.0, "Returned", "Credit Card"),
    ("ORD007", "CUST005", date(2024, 4, 8), "Electronics", 1, 680.0, "Completed", "Bank Transfer"),
    ("ORD008", "CUST003", date(2024, 4, 15), "Books", 2, 25.0, "Completed", "PayPal"),
    ("ORD009", "CUST006", date(2024, 5, 20), "Clothing", 1, 420.0, "Pending", "Credit Card"),
    ("ORD010", "CUST004", date(2024, 6, 10), "Home & Garden", 3, 95.0, "Cancelled", "Bank Transfer")
]

# Create DataFrame with schema
df = spark.createDataFrame(data, schema)

# Add calculated column for total order value (quantity * unit_price)
df = df.withColumn("order_value", col("quantity") * col("unit_price"))

print("Sample Data:")
df.show()



# TECHNICAL SCENARIO: Product-based revenue tier analysis using conditional aggregation
# Challenge: Categorize product revenue into tiers (High >$1000, Medium $200-1000, Low <$200)
# Solution: Group by product_category, sum order_value, then apply conditional logic for tiering
print("\nQ1. REVENUE CATEGORIZATION BY PRODUCT CATEGORY")

# Calculate total revenue per product category first
product_revenue = df.groupBy("product_category").agg(
    spark_sum("order_value").alias("total_revenue"),
    count("*").alias("order_count")
)

# Apply conditional categorization based on total revenue thresholds
revenue_by_product = product_revenue.withColumn(
    "revenue_tier",
    when(col("total_revenue") > 1000, "High")
    .when(col("total_revenue").between(200, 1000), "Medium")
    .otherwise("Low")
).select(
    "product_category",
    "total_revenue",
    "order_count",
    "revenue_tier"
).orderBy(col("total_revenue").desc())

print("Revenue Categorization by Product Category:")
revenue_by_product.show()


# TECHNICAL SCENARIO: Weighted scoring system using conditional aggregation
# Challenge: Assign different weights to order statuses and calculate performance metrics
# Solution: Use when().otherwise() with numeric weights in aggregation functions
print("\nQ2. PAYMENT METHOD PERFORMANCE SCORE")

# Calculate weighted performance score using conditional logic
performance_score = df.groupBy("payment_method").agg(
    spark_sum(
        when(col("order_status") == "Completed", 2)    # +2 for completed
        .when(col("order_status") == "Pending", 1)     # +1 for pending
        .when(col("order_status") == "Cancelled", -1)  # -1 for cancelled
        .otherwise(0)                                  # 0 for other statuses
    ).alias("performance_score"),
    count("*").alias("total_orders")
).orderBy(col("performance_score").desc())

print("Performance Score by Payment Method:")
performance_score.show()

# TECHNICAL SCENARIO: Conditional counting and percentage calculation
# Challenge: Calculate completion rate as percentage of total orders per group
# Solution: Use conditional sum with when() for numerator, count() for denominator
print("\nQ3. SUCCESS RATE BY PAYMENT METHOD")

# Calculate completion rate using conditional aggregation
success_rate = df.groupBy("payment_method").agg(
    # Count completed orders
    spark_sum(when(col("order_status") == "Completed", 1).otherwise(0)).alias("completed_orders"),
    count("*").alias("total_orders")
).withColumn(
    # Calculate percentage completion rate
    "completion_rate_pct",
    (col("completed_orders") / col("total_orders") * 100).cast("decimal(5,2)")
).orderBy(col("completion_rate_pct").desc())

print("Success Rate by Payment Method:")
success_rate.show()

# TECHNICAL SCENARIO: Customer repeat purchase behavior using conditional time-based analysis
# Challenge: Identify customers with repeat purchases vs one-time buyers and their value contribution
# Solution: Count orders per customer, classify behavior type, aggregate business metrics
print("\nQ4. CUSTOMER  PURCHASE BEHAVIOR ANALYSIS")

# Analyze customer purchase frequency and classify behavior
customer_behavior = df.groupBy("customer_id").agg(
    # Count total orders per customer
    count("*").alias("order_count"),

    # Calculate total customer value
    spark_sum("order_value").alias("total_customer_value"),

    # Count successful orders (completed only)
    spark_sum(
        when(col("order_status") == "Completed", 1).otherwise(0)
    ).alias("successful_orders")
).withColumn(
    # Classify customer behavior type
    "customer_behavior",
    when(col("order_count") >= 3, "Frequent Buyer")
    .when(col("order_count") == 2, "Repeat Buyer")
    .otherwise("One-Time Buyer")
).withColumn(
    # Calculate success rate per customer
    "success_rate_pct",
    (col("successful_orders") / col("order_count") * 100).cast("decimal(5,2)")
)

print("Customer Repeat Purchase Behavior:")
customer_behavior.orderBy (col("customer_id")).show()

# TECHNICAL SCENARIO: Complex conditional revenue analysis with multiple status handling
# Challenge: Categorize revenue streams - Earned (Completed), Lost (Cancelled+Returned), Pending (Pending)
# Solution: Use multiple conditional sums to separate revenue by status, calculate retention metrics
print("\nQ5. QUARTERLY PERFORMANCE WITH COMPREHENSIVE STATUS BREAKDOWN")

# Extract quarter and year, then calculate conditional revenue across all statuses
quarterly_performance = df.withColumn("quarter", quarter("order_date")) \
                         .withColumn("year", year("order_date")) \
                         .groupBy("year", "quarter").agg(
    # Earned revenue from completed orders only
    spark_sum(
        when(col("order_status") == "Completed", col("order_value")).otherwise(0)
    ).alias("earned_revenue"),
    # Lost revenue from cancelled AND returned orders
    spark_sum(
        when(col("order_status").isin(["Cancelled", "Returned"]), col("order_value")).otherwise(0)
    ).alias("lost_revenue"),
    # Pending revenue from orders still in process
    spark_sum(
        when(col("order_status") == "Pending", col("order_value")).otherwise(0)
    ).alias("pending_revenue"),
    # Total potential revenue across all orders
    spark_sum("order_value").alias("total_potential_revenue"),
    # Order count breakdown by status
    spark_sum(when(col("order_status") == "Completed", 1).otherwise(0)).alias("completed_orders"),
    spark_sum(when(col("order_status").isin(["Cancelled", "Returned"]), 1).otherwise(0)).alias("lost_orders"),
    spark_sum(when(col("order_status") == "Pending", 1).otherwise(0)).alias("pending_orders"),
    count("*").alias("total_orders")
).orderBy("year", "quarter")

print("Comprehensive Quarterly Performance Analysis:")
quarterly_performance.show()


# Clean up
spark.stop()


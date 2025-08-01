from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count, min as spark_min, \
    datediff, date_format, lag, coalesce, lit, row_number,
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from datetime import datetime, date

# Initialize Spark session
spark = SparkSession.builder \
    .appName("EcommerceOrderAnalytics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Create dummy datasets for demonstration
print("📊 Creating dummy e-commerce datasets...")

# Orders table - Contains order-level information
orders_data = [
    (1, 101, date(2024, 1, 15), "completed", 5.99),
    (2, 102, date(2024, 1, 20), "completed", 8.99),
    (3, 103, date(2024, 2, 10), "completed", 12.99),
    (4, 101, date(2024, 2, 25), "completed", 5.99),
    (5, 104, date(2024, 3, 5), "completed", 15.99),
    (6, 105, date(2024, 3, 12), "completed", 10.99),
    (7, 102, date(2024, 3, 18), "completed", 8.99),
    (8, 106, date(2024, 4, 2), "completed", 7.99),
]

orders_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("order_date", DateType(), True),
    StructField("order_status", StringType(), True),
    StructField("shipping_cost", DoubleType(), True)
])

orders = spark.createDataFrame(orders_data, orders_schema)

# Order items table - Contains product-level details for each order
order_items_data = [
    (1, 201, 2, 25.99, 2.00),
    (1, 202, 1, 45.99, 5.00),
    (2, 203, 3, 15.99, 1.50),
    (3, 201, 1, 25.99, 0.00),
    (3, 204, 2, 35.99, 3.00),
    (4, 202, 1, 45.99, 5.00),
    (5, 205, 1, 89.99, 10.00),
    (6, 203, 2, 15.99, 1.50),
    (7, 204, 3, 35.99, 3.00),
    (8, 201, 1, 25.99, 0.00),
]

order_items_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("discount", DoubleType(), True)
])

order_items = spark.createDataFrame(order_items_data, order_items_schema)

# Products table - Contains product catalog information
products_data = [
    (201, "Wireless Headphones", "Electronics", "TechBrand", 15.00),
    (202, "Gaming Keyboard", "Electronics", "GameCorp", 25.00),
    (203, "Coffee Mug", "Home", "KitchenPro", 8.00),
    (204, "Desk Lamp", "Home", "HomeBrand", 20.00),
    (205, "Laptop Stand", "Electronics", "TechBrand", 45.00),
]

products_schema = StructType([
    StructField("product_id", IntegerType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("cost_price", DoubleType(), True)
])

products = spark.createDataFrame(products_data, products_schema)

# Customers table - Contains customer information and tier classification
customers_data = [
    (101, "Alice Johnson", "Seattle", "WA", date(2023, 12, 1), "Premium"),
    (102, "Bob Smith", "Portland", "OR", date(2023, 11, 15), "Standard"),
    (103, "Carol Davis", "San Francisco", "CA", date(2024, 1, 5), "Premium"),
    (104, "David Wilson", "Los Angeles", "CA", date(2024, 2, 10), "Standard"),
    (105, "Eva Brown", "Seattle", "WA", date(2024, 2, 20), "Premium"),
    (106, "Frank Miller", "Portland", "OR", date(2024, 3, 1), "Standard"),
]

customers_schema = StructType([
    StructField("customer_id", IntegerType(), True),
    StructField("customer_name", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("registration_date", DateType(), True),
    StructField("customer_tier", StringType(), True)
])

customers = spark.createDataFrame(customers_data, customers_schema)

print("Dummy datasets created successfully!")

# REQUIREMENT 1: Calculate total revenue, profit margin, and average order value by customer tier and product category
print("\n📈 Requirement 1: Revenue, Profit Margin & AOV by Customer Tier and Product Category")

# Join all tables to get complete order information
# This creates a comprehensive view combining order, item, product, and customer data
complete_orders = orders.join(order_items, "order_id") \
    .join(products, "product_id") \
    .join(customers, "customer_id")

# Calculate revenue and profit for each order item
# Revenue = (unit_price - discount) * quantity
# Profit = revenue - (cost_price * quantity)
revenue_profit = complete_orders.withColumn(
    "revenue", (col("unit_price") - col("discount")) * col("quantity")
).withColumn(
    "profit", ((col("unit_price") - col("discount")) * col("quantity")) - (col("cost_price") * col("quantity"))
)

# Aggregate metrics by customer tier and product category
tier_category_metrics = revenue_profit.groupBy("customer_tier", "category") \
    .agg(
        spark_sum("revenue").alias("total_revenue"),
        spark_sum("profit").alias("total_profit"),
        count("order_id").alias("total_orders")
    ) \
    .withColumn("profit_margin", col("total_profit") / col("total_revenue") * 100) \
    .withColumn("avg_order_value", col("total_revenue") / col("total_orders"))

print("Customer Tier & Category Metrics:")
tier_category_metrics.show(truncate=False)

# REQUIREMENT 2: Find top 5 customers by lifetime value in each state
print("\n🏆 Requirement 2: Top 5 Customers by Lifetime Value in Each State")

# Calculate customer lifetime value (CLV) - total revenue generated by each customer
customer_ltv = revenue_profit.groupBy("customer_id", "customer_name", "state") \
    .agg(spark_sum("revenue").alias("lifetime_value"))

# Use window function to rank customers within each state
# row_number() assigns unique ranks, while rank() can have ties
state_window = Window.partitionBy("state").orderBy(col("lifetime_value").desc())
top_customers_by_state = customer_ltv.withColumn(
    "rank", row_number().over(state_window)
).filter(col("rank") <= 5)

print("Top 5 Customers by State:")
top_customers_by_state.show(truncate=False)

# REQUIREMENT 3: Calculate month-over-month growth in orders and revenue by brand
print("\n📊 Requirement 3: Month-over-Month Growth by Brand")

# Extract year-month from order date for monthly aggregation
monthly_brand_metrics = revenue_profit.withColumn(
    "year_month", date_format(col("order_date"), "yyyy-MM")
) \
.groupBy("brand", "year_month") \
.agg(
    count("order_id").alias("monthly_orders"),
    spark_sum("revenue").alias("monthly_revenue")
) \
.orderBy("brand", "year_month")

# Calculate month-over-month growth using lag window function
# lag() gets the previous row's value within the same brand partition
brand_window = Window.partitionBy("brand").orderBy("year_month")
mom_growth = monthly_brand_metrics.withColumn(
    "prev_month_orders", lag("monthly_orders", 1).over(brand_window)
).withColumn(
    "prev_month_revenue", lag("monthly_revenue", 1).over(brand_window)
).withColumn(
    "orders_growth_pct",
    ((col("monthly_orders") - col("prev_month_orders")) / col("prev_month_orders") * 100)
).withColumn(
    "revenue_growth_pct",
    ((col("monthly_revenue") - col("prev_month_revenue")) / col("prev_month_revenue") * 100)
)

print("Month-over-Month Growth by Brand:")
mom_growth.show(truncate=False)

# REQUIREMENT 4: Determine most profitable product categories for premium customers
print("\n💎 Requirement 4: Most Profitable Categories for Premium Customers")

# Filter for premium customers only and calculate category profitability
premium_category_profit = revenue_profit.filter(col("customer_tier") == "Premium") \
    .groupBy("category") \
    .agg(
        spark_sum("profit").alias("total_profit"),
        spark_sum("revenue").alias("total_revenue"),
        count("order_id").alias("order_count")
    ) \
    .withColumn("profit_margin", col("total_profit") / col("total_revenue") * 100) \
    .orderBy(col("total_profit").desc())

print("Most Profitable Categories for Premium Customers:")
premium_category_profit.show(truncate=False)

# REQUIREMENT 5: Calculate customer acquisition metrics
print("\n🎯 Requirement 5: Customer Acquisition Metrics by State")

# Find each customer's first order date
first_orders = orders.groupBy("customer_id") \
    .agg(spark_min("order_date").alias("first_order_date"))

# Calculate days between registration and first order
acquisition_metrics = customers.join(first_orders, "customer_id") \
    .withColumn(
        "days_to_first_order",
        datediff(col("first_order_date"), col("registration_date"))
    ) \
    .groupBy("state") \
    .agg(
        avg("days_to_first_order").alias("avg_days_to_first_order"),
        count("customer_id").alias("total_customers")
    )

print("Customer Acquisition Metrics by State:")
acquisition_metrics.show(truncate=False)

print("\n🎉 E-commerce Order Analytics Complete!")
print("Key insights generated for business decision-making.")

# Clean up Spark session
spark.stop()
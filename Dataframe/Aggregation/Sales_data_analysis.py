from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark Session - Entry point for all Spark operations
spark = SparkSession.builder \
    .appName("SalesPerformanceAnalysis") \
    .getOrCreate()

# Create dummy sales dataset for demonstration
sales_data = [
    ("S001", "John", "Electronics", "Laptop", 1200.00, "2024-01-15", "North"),
    ("S002", "Sarah", "Clothing", "Jacket", 80.00, "2024-01-16", "South"),
    ("S003", "Mike", "Electronics", "Phone", 800.00, "2024-01-17", "East"),
    ("S004", "John", "Home", "Chair", 150.00, "2024-01-18", "North"),
    ("S005", "Sarah", "Electronics", "Tablet", 300.00, "2024-01-19", "South"),
    ("S006", "Mike", "Clothing", "Shoes", 120.00, "2024-01-20", "East"),
    ("S007", "John", "Electronics", "Watch", 250.00, "2024-01-21", "North"),
    ("S008", "Lisa", "Home", "Table", 400.00, "2024-01-22", "West"),
    ("S009", "Sarah", "Electronics", "Headphones", 100.00, "2024-01-23", "South"),
    ("S010", "Mike", "Home", "Lamp", 75.00, "2024-01-24", "East")
]

# Define schema for type safety and performance optimization
schema = StructType([
    StructField("sale_id", StringType(), True),
    StructField("salesperson", StringType(), True),
    StructField("category", StringType(), True),
    StructField("product", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("sale_date", StringType(), True),
    StructField("region", StringType(), True)
])

# Create DataFrame with explicit schema
df = spark.createDataFrame(sales_data, schema)

# Data preprocessing - Convert string date to proper date type
df = df.withColumn("sale_date", to_date(col("sale_date"), "yyyy-MM-dd"))

print("Original Sales Data:")
df.show()

# 1. SALESPERSON PERFORMANCE ANALYSIS
"""
1. Salesperson Performance Analysis
• Applied sum("amount") → Total revenue per salesperson
• Used count("sale_id") → Number of transactions per person
• Calculated avg("amount") → Average deal size per person
• Implemented max("amount") → Highest single sale per person
• Applied min("amount") → Lowest single sale per person
• Used countDistinct("category") → Number of unique categories sold by each person
• Rounded average sale value to 2 decimal places for precision
"""
print("Top Performers by Total Sales:")
salesperson_performance = df.groupBy("salesperson") \
    .agg(
        sum("amount").alias("total_sales"),               # Total revenue per salesperson
        count("sale_id").alias("total_transactions"),     # Number of transactions per person
        avg("amount").alias("avg_sale_value"),            #Average deal size per person
        max("amount").alias("highest_sale"),
        min("amount").alias("lowest_sale"),
        countDistinct("category").alias("categories_sold")
    ) \
    .withColumn("avg_sale_value", round(col("avg_sale_value"), 2)) \
    .orderBy(desc("total_sales"))  # Sort by highest revenue first

salesperson_performance.show()

# 2. CATEGORY PERFORMANCE ANALYSIS
"""
• Calculated total revenue using df.agg(sum("amount"))
• Applied sum("amount") → Total revenue per category
• Used count("*") → Number of units sold per category
• Calculated avg("amount") → Average price point per category
• Applied countDistinct("salesperson") → Number of salespeople selling in each category
• Implemented revenue share calculation: (category_revenue / total_revenue) * 100
• Rounded results to 2 decimal places
"""
print("Product Category Analysis:")
total_revenue = df.agg(sum("amount")).collect()[0][0]
category_analysis = df.groupBy("category") \
    .agg(
        sum("amount").alias("category_revenue"),
        count("*").alias("units_sold"),
        avg("amount").alias("avg_price_point"),
        countDistinct("salesperson").alias("salespeople_count")

    ) \
    .withColumn("revenue_share_pct",
                round((col("category_revenue") / total_revenue) * 100, 2)
    ) \
    .orderBy(desc("category_revenue"))

category_analysis.show()

# 3. REGIONAL PERFORMANCE ANALYSIS
print("Regional Sales Distribution:")
regional_performance = df.groupBy("region") \
    .agg(
        sum("amount").alias("regional_sales"),
        count("*").alias("total_sales_count"),
        countDistinct("salesperson").alias("active_salespeople"),
        avg("amount").alias("avg_regional_sale"),
        countDistinct("category").alias("categories_available")
    ) \
    .withColumn("avg_regional_sale", round(col("avg_regional_sale"), 2)) \
    .orderBy(desc("regional_sales"))

regional_performance.show()

# 4. TIME-BASED ANALYSIS
print("Daily Sales Trend:")
daily_sales = df.groupBy("sale_date") \
    .agg(
        sum("amount").alias("daily_revenue"),
        count("*").alias("daily_transactions"),
       countDistinct("salesperson").alias("active_salespeople")
    ) \
    .withColumn("avg_transaction_value",
                round(col("daily_revenue") / col("daily_transactions"), 2)) \
    .orderBy("sale_date")

daily_sales.show()

# 5. ADVANCED INSIGHTS - Window Functions for Ranking (TOP PERFORMERS BY CATEGORY)
print("Advanced Analytics - Sales Ranking by Category:")
from pyspark.sql.window import Window

# Create window specification for ranking within each category
window_spec = Window.partitionBy("category").orderBy(desc("amount"))

top_sales_by_category = df.withColumn("rank_in_category",
                                                 row_number().over(window_spec)) \
    .filter(col("rank_in_category") <= 2) \
    .select("category", "salesperson", "product", "amount", "rank_in_category") \
    .orderBy("category", "rank_in_category")

top_sales_by_category.show()

# 6. PERFORMANCE INSIGHTS WITH CONDITIONAL LOGIC
print("\n6.6 Sales Performance Tier Classification:")
performance_tiers = salesperson_performance \
    .withColumn("performance_tier",
        when(col("total_sales") >= 1200, " Top Performer")
        .when(col("total_sales") >= 800, " High Performer")
        .when(col("total_sales") >= 400, " Medium Performer")
        .otherwise("Developing")
    ) \
    .withColumn("efficiency_score",
        round(col("total_sales") / col("total_transactions"), 2)
    )

performance_tiers.select("salesperson", "total_sales", "total_transactions",
                        "efficiency_score", "performance_tier").show()
# Clean up resources
spark.stop()


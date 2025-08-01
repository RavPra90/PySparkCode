from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, desc, row_number, sum

# Create Spark session
spark = SparkSession.builder.appName("TopSalesAnalysis").getOrCreate()

# 📊 Create realistic sales dummy data
sales_data = [
    ("North", "Alice Johnson", 125000),
    ("North", "Bob Smith", 98000),
    ("North", "Carol Davis", 142000),     # Top performer North
    ("South", "Emma Brown", 156000),      # Top performer South
    ("South", "Frank Miller", 91000),
    ("South", "Grace Lee", 134000),
    ("East", "Ivy Chen", 118000),
    ("East", "Jack Robinson", 145000),    # Top performer East
    ("East", "Kelly Martinez", 89000),
    ("West", "Liam Anderson", 167000),    # Top performer West
    ("West", "Maya Patel", 94000),
    ("West", "Noah Garcia", 128000)
]

# Create DataFrame with schema
df = spark.createDataFrame(sales_data, ["region", "salesperson", "revenue"])

print("Original Sales Data:")
df.orderBy("region", desc("revenue")).show()

# STEP 1: Define Window Specification
# This tells Spark HOW to group and sort our data
window_spec = Window.partitionBy("region").orderBy(desc("revenue"))

# partitionBy("region") → Creates separate groups for each region
# orderBy(desc("revenue")) → Sorts each region's data by revenue (highest first)

#STEP 2: Apply row_number() window function
# This assigns rank 1, 2, 3... to each salesperson within their region
df_with_rank = df.withColumn("rank", row_number().over(window_spec))


print("Sales Data with Rankings:")
df_with_rank.orderBy("region", "rank").show()

# STEP 3: Filter for Top 2 performers per region
top_3_per_region = df_with_rank.filter(col("rank") <= 2)
# filter(col("rank") <= 2) → Keeps only rows where rank is 1 or 2

print("TOP 2 PERFORMERS BY REGION:")
top_3_per_region.orderBy("region", "rank").show()

#Let's also calculate total revenue per region for context
region_totals = df.groupBy("region").agg(sum("revenue").alias("total_revenue"))
print("Total Revenue by Region:")
region_totals.show()

spark.stop()

"""
📚 STEP-BY-STEP APPROACH:
1️⃣ CREATE WINDOW SPEC: Define how to partition (group) and order data
2️⃣ APPLY RANKING: Use row_number() to assign ranks within each partition  
3️⃣ FILTER RESULTS: Keep only top N records using simple filter condition



🚀 PERFORMANCE OPTIMIZATION:
• Partition your data appropriately before window operations
• Use broadcast joins if dimension tables are small
• Consider bucketing for frequently queried columns
• Monitor Spark UI for skewed partitions

🔄 ALTERNATIVE APPROACHES:
• Use SQL: SELECT *, ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) as rank
• Use DataFrame API with window functions (shown above)
• Avoid collect() + Python sorting (inefficient for big data)
"""
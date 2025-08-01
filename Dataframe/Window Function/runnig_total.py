from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Initialize Spark session
spark = SparkSession.builder.appName("WindowAnalytics").getOrCreate()

# Create diversified sales dataset
# Create dummy sales dataset with diverse scenarios
sales_data = [
    ("2024-01-01", "Store_A", "Product_X", "North", 1000),
    ("2024-01-01", "Store_B", "Product_Y", "South", 1500),
    ("2024-01-02", "Store_A", "Product_X", "North", 1200),
    ("2024-01-02", "Store_B", "Product_Y", "South", 800),
    ("2024-01-03", "Store_A", "Product_Z", "North", 900),
    ("2024-01-03", "Store_B", "Product_X", "South", 1100),
    ("2024-01-04", "Store_A", "Product_Y", "North", 1300),
    ("2024-01-04", "Store_B", "Product_Z", "South", 950),
    ("2024-01-05", "Store_A", "Product_X", "North", 1400),
    ("2024-01-05", "Store_B", "Product_Y", "South", 1250),
    ("2024-01-08", "Store_A", "Product_Z", "North", 1600),  # Week 2
    ("2024-01-08", "Store_B", "Product_X", "South", 1350),
    ("2024-01-09", "Store_A", "Product_Y", "North", 1150),
    ("2024-01-09", "Store_B", "Product_Z", "South", 1050),
    ("2024-02-01", "Store_A", "Product_X", "North", 1700),  # Month 2
    ("2024-02-01", "Store_B", "Product_Y", "South", 1450),
]

# Define schema for better performance
schema = StructType([
    StructField("date", StringType(), True),
    StructField("store", StringType(), True),
    StructField("product", StringType(), True),
    StructField("region", StringType(), True),
    StructField("sales_amount", IntegerType(), True)
])

# Create DataFrame with proper date formatting and week calculation
df = spark.createDataFrame(sales_data, schema)
df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
df = df.withColumn("month", date_format(col("date"), "yyyy-MM"))
# Calculate week number from date (ISO week format)
df = df.withColumn("week_num", weekofyear(col("date")))

print("=== Sample Dataset ===")
df.show()

#Scenario 1: Daily Cumulative (Running) Sum per Store
#==================================================================
print("=== 1. Daily Cumulative (Running) Sum per Store ===")
# Key Steps:
# 1. Partition data by store (separate calculation per store)
# 2. Order by date for chronological processing
# 3. Use unbounded preceding to include all previous rows

# Define window: Partition by store, order by date, unbounded preceding to current row
window_spec = Window.partitionBy("store").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

running_total = df.withColumn("running_total", sum("sales_amount").over(window_spec))
running_total.select("date", "store", "sales_amount", "running_total").show()


#Scenario 2: 3-Day Rolling Sum of Sales for Each Product
#===============================================================================================
# Key Steps:
# 1. Partition by product for product-wise calculations
# 2. Use rowsBetween(-2, 0) for 3-day sliding window
# 3. Current row (0) + 2 preceding rows (-2) = 3-day sum

print("=== 2. 3-Day Rolling Sum of Sales for Each Product ===")
# Rolling window: 2 rows before + current row = 3 days total
rolling_window = Window.partitionBy("product").orderBy("date").rowsBetween(-2, 0)

rolling_sum = df.withColumn("rolling_3day_sum", sum("sales_amount").over(rolling_window))
rolling_sum.select("date", "product", "sales_amount", "rolling_3day_sum").show()


#Scenario 3. Rank Top-Selling Product per Day using Window Function
#=====================================================================================================
print("\n=== 3. DAILY PRODUCT RANKING ===")
# Key Steps:
# 1. Partition data by date (separate ranking per day)
# 2. Order by sales_amount descending for highest-to-lowest ranking
# 3. Use row_number() to assign unique ranks without ties
window_rank = Window.partitionBy("date").orderBy(desc("sales_amount"))

df_ranked = df.withColumn(
    "daily_rank",
    row_number().over(window_rank)  # Assign rank 1 to highest sales
).select("date", "product", "sales_amount", "daily_rank")

df_ranked.orderBy("date", "daily_rank").show()


#Scenario 4: Running Total of Monthly Revenue by Region
#===================================================================================================
print("=== 4. Running Total of Monthly Revenue by Region ===")
# Key Steps:
# 1. First aggregate data by month and region to get monthly totals
# 2. Partition data by region (separate calculation per region)
# 3. Order by month for chronological processing
# 4. Use unbounded preceding to current row for cumulative sum across months

# Aggregate by month and region first
monthly_sales = df.groupBy("month", "region") \
    .agg(sum("sales_amount").alias("monthly_revenue"))

# Calculate running total across months per region
monthly_window = Window.partitionBy("region").orderBy("month").rowsBetween(Window.unboundedPreceding, Window.currentRow)

monthly_running = monthly_sales.withColumn("running_monthly_total", sum("monthly_revenue").over(monthly_window))
monthly_running.show()


#Scenario 5: Week-over-Week Sales Growth Percentage by Store
#=================================================================================================
print("\n=== 5. WEEK-OVER-WEEK GROWTH BY STORE ===")
# Key Steps:
# 1. First aggregate data by week and store to get weekly totals
# 2. Partition data by store (separate calculation per store)
# 3. Order by week_num for chronological processing
# 4. Use lag(1) to get previous week's value for comparison
# 5. Calculate percentage growth formula: ((current - previous) / previous) * 100

# Note: week_num is calculated from date using weekofyear() function
weekly_sales = df.groupBy("week_num", "store") \
    .agg(sum("sales_amount").alias("weekly_sales"))

window_weekly = Window.partitionBy("store").orderBy("week_num")

df_growth = weekly_sales.withColumn(
    "prev_week_sales",
    lag("weekly_sales", 1).over(window_weekly)  # Get previous week's sales
).withColumn(
    "week_over_week_growth_pct",
    # Calculate percentage growth: ((current - previous) / previous) * 100
    when(col("prev_week_sales").isNotNull(),
         ((col("weekly_sales") - col("prev_week_sales")) / col("prev_week_sales") * 100)
    ).otherwise(None)
)

df_growth.select("week_num", "store", "weekly_sales",
                "prev_week_sales", "week_over_week_growth_pct").show()

# Clean up
spark.stop()
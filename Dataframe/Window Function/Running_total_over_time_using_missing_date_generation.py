"""
**Scenario:** Calculate running totals of daily orders per store, handling date gaps gracefully
**Challenge:** Missing dates (zero order days) need to be filled for accurate running totals
**Solution:** Window functions + date range generation for complete time-series analysis
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Initialize Spark session
spark = SparkSession.builder.appName("RunningTotalDemo").getOrCreate()

# Create dummy dataset with realistic gaps (some stores have missing dates)
data = [
    ("STORE_001", "2024-01-01", 15),
    ("STORE_001", "2024-01-02", 22),
    ("STORE_001", "2024-01-04", 18),  # Missing Jan 3rd
    ("STORE_001", "2024-01-05", 25),
    ("STORE_002", "2024-01-01", 30),
    ("STORE_002", "2024-01-03", 12),  # Missing Jan 2nd
    ("STORE_002", "2024-01-05", 28),  # Missing Jan 4th
]

# Define schema for better performance and type safety
schema = StructType([
    StructField("store_id", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("orders", IntegerType(), True)
])

# Create DataFrame and convert string dates to proper date type
df = spark.createDataFrame(data, schema)
df = df.withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))

print("Original Data with Date Gaps:")
df.orderBy("store_id", "order_date").show()

# STEP 1: Generate complete date range
# Create a subquery to get min/max dates per store (more efficient approach)
store_date_ranges = df.groupBy("store_id").agg(
    min("order_date").alias("min_date"),
    max("order_date").alias("max_date")
)

print("Date Ranges per Store:")
store_date_ranges.show()

# Generate complete date sequence for each store individually
complete_dates_per_store = store_date_ranges.select(
    col("store_id"),
    explode(
        expr("sequence(min_date, max_date, interval 1 day)")
    ).alias("order_date")
)

print("Complete Date Sequence per Store (sample):")
complete_dates_per_store.orderBy("store_id", "order_date").show(10)

#STEP 2: Left join original data with complete date sequence
complete_data = complete_dates_per_store.join(
    df,
    on=["store_id", "order_date"],
    how="left"  # Keep all dates, even if no orders
).select(
    "store_id",
    "order_date",
    coalesce(col("orders"), lit(0)).alias("daily_orders")  # Fill gaps with 0
)

print("Complete Data with Zero-Filled Gaps:")
complete_data.orderBy("store_id", "order_date").show()

# 🏃‍♂️ STEP 3: Calculate running total using Window function
# Define window: partition by store, order by date, unbounded preceding to current row
window_spec = Window.partitionBy("store_id") \
    .orderBy("order_date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)


# Apply sum over window to get running total
result = complete_data.withColumn(
    "running_total",
    sum("daily_orders").over(window_spec)  # Cumulative sum within each store
)

print("Final Result - Running Totals with Gap Handling:")
result.orderBy("store_id", "order_date").show()


# Clean up
spark.stop()

"""
🎯 KEY INSIGHTS:
✅ Date gaps automatically filled with zero orders
✅ Running totals remain accurate across missing dates  
✅ Each store maintains independent running total
✅ Window functions provide efficient computation

⚠️ EDGE CASES COVERED:
• Missing dates in time series → Zero-filled
• Different stores with different date ranges → Cross join handles it
• Empty datasets → Coalesce prevents null errors
• Single-day data → Window still works correctly

💡 PERFORMANCE TIPS:
• Use date partitioning in production for large datasets
• Consider caching intermediate results for complex transformations
• Broadcast small lookup tables for better join performance
• Use appropriate window frame (rows vs range) based on your needs
"""



"""
🔧 STEP-BY-STEP APPROACH TO AVOID PERFORMANCE PITFALLS:

**PROBLEM:** Calculate running totals with date gap handling efficiently

**SOLUTION STRATEGY:**

1️⃣ **Avoid collect() - Stay Distributed:**
   - Use groupBy per store to get min/max dates
   - Keep everything as DataFrame operations
   - Let Spark's optimizer handle the execution plan

2️⃣ **Eliminate Cross Join - Generate Dates Smarter:**
   - Generate date sequences per store individually  
   - Use expr("sequence(min_date, max_date, interval 1 day)")
   - This creates only the dates each store actually needs

3️⃣ **Efficient Join Strategy:**
   - Left join original data with generated date sequences
   - Spark can optimize this much better than cross joins
   - Reduces data shuffling significantly

4️⃣ **Alternative Approaches for Very Large Data:**

   **Option A: Window with Forward Fill**
   ```python
   # Instead of generating missing dates, use lag/lead to forward fill
   window_fill = Window.partitionBy("store_id").orderBy("order_date")
   filled_data = df.withColumn("prev_orders", 
                              last("orders", True).over(window_fill))
   ```

   **Option B: Spark SQL Approach**  
   ```python
   df.createOrReplaceTempView("orders")
   result = spark.sql('''
       WITH date_series AS (
           SELECT store_id, 
                  explode(sequence(min(order_date), max(order_date), interval 1 day)) as order_date
           FROM orders GROUP BY store_id
       )
       SELECT d.store_id, d.order_date,
              COALESCE(o.orders, 0) as daily_orders,
              SUM(COALESCE(o.orders, 0)) OVER (
                  PARTITION BY d.store_id 
                  ORDER BY d.order_date 
                  ROWS UNBOUNDED PRECEDING
              ) as running_total
       FROM date_series d
       LEFT JOIN orders o ON d.store_id = o.store_id AND d.order_date = o.order_date
   ''')
   ```

   **Option C: Delta Lake Time Travel (Production Recommendation)**
   ```python
   # For incremental updates in production
   # Only process new dates, merge with existing running totals
   # Use Delta Lake's MERGE operation for upserts
   ```

**WHEN TO USE EACH APPROACH:**
- Small datasets (< 1M rows): Any approach works
- Medium datasets (1M - 100M rows): Use improved approach  
- Large datasets (> 100M rows): Consider Option B or C
- Streaming data: Use structured streaming with watermarks
"""
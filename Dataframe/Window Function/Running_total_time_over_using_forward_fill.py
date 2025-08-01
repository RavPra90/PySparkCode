# PySpark: Running Total Over Time with Forward Fill (No Missing Date Generation)
# Efficient approach that works with existing data points only

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("RunningTotalForwardFill").getOrCreate()

# STEP 1: Create sample dataset with missing dates
data = [
    ("STORE_001", "2024-01-01", 15),
    ("STORE_001", "2024-01-02", 22),
    ("STORE_001", "2024-01-04", 18),  # Missing Jan 3rd
    ("STORE_001", "2024-01-05", 25),
    ("STORE_002", "2024-01-01", 30),
    ("STORE_002", "2024-01-03", 12),  # Missing Jan 2nd
    ("STORE_002", "2024-01-05", 28),  # Missing Jan 4th
]

schema = StructType([
    StructField("store_id", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("orders", IntegerType(), True)
])

# Create DataFrame and convert string dates to proper date type
df_orders = spark.createDataFrame(data, schema)
df_orders = df_orders.withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))

print("Original Dataset with Date Gaps:")
df_orders.orderBy("store_id", "order_date").show()

# STEP 2: Forward fill missing values using window functions
# Window specification: partition by store, order by date, from start to current row
forward_fill_window = Window.partitionBy("store_id") \
    .orderBy("order_date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Apply forward fill: use last non-null value within the window
# If no previous value exists, default to 0
df_forward_filled = df_orders.withColumn(
    "orders_filled",
    coalesce(
        col("orders"),  # Use current value if exists
        last(col("orders"), ignorenulls=True).over(forward_fill_window),  # Use last known value
        lit(0)  # Default to 0 if no previous data
    )
)

print("After Forward Fill (conceptual - gaps handled logically):")
df_forward_filled.select("store_id", "order_date", "orders", "orders_filled") \
    .orderBy("store_id", "order_date").show()

# STEP 3: Calculate running total using the forward-filled values
# Window for cumulative sum: partition by store, order by date
running_total_window = Window.partitionBy("store_id") \
    .orderBy("order_date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Compute running total of forward-filled orders
df_final = df_forward_filled.withColumn(
    "running_total",
    sum("orders_filled").over(running_total_window)
)

print("FINAL RESULT: Running Total with Forward Fill Logic")
df_final.select("store_id", "order_date", "orders", "orders_filled", "running_total") \
    .orderBy("store_id", "order_date").show()

# STEP 4: Add gap detection for analysis (optional)
gap_detection_window = Window.partitionBy("store_id").orderBy("order_date")

df_with_analysis = df_final.withColumn(
    "prev_date",
    lag(col("order_date"), 1).over(gap_detection_window)
).withColumn(
    "days_since_last",
    datediff(col("order_date"), col("prev_date"))
).withColumn(
    "has_date_gap",
    when(col("days_since_last") > 1, True).otherwise(False)
)

print("Gap Analysis (shows where forward fill logic was applied):")
df_with_analysis.select(
    "store_id", "order_date", "orders", "days_since_last",
    "has_date_gap", "running_total"
).orderBy("store_id", "order_date").show()

# Clean up
spark.stop()

# DETAILED EXPLANATION OF APPROACH:

"""
FORWARD FILL CONCEPT:
Forward fill means using the last known value to handle missing data points.
Instead of creating explicit rows for missing dates, we logically carry forward
the previous value when calculating aggregations.

𝗖𝗼𝗿𝗲 𝗧𝗲𝗰𝗵𝗻𝗶𝗰𝗮𝗹 𝗖𝗼𝗻𝗰𝗲𝗽𝘁𝘀: 

𝗙𝗼𝗿𝘄𝗮𝗿𝗱 𝗙𝗶𝗹𝗹 𝗟𝗼𝗴𝗶𝗰 - Instead of generating missing date rows, we carry forward the last known value using `𝙡𝙖𝙨𝙩()` with `𝙞𝙜𝙣𝙤𝙧𝙚𝙣𝙪𝙡𝙡𝙨=𝙏𝙧𝙪𝙚` during aggregation
𝗦𝗺𝗮𝗿𝘁 𝗪𝗶𝗻𝗱𝗼𝘄 𝗙𝘂𝗻𝗰𝘁𝗶𝗼𝗻𝘀 - Using `𝙧𝙤𝙬𝙨𝘽𝙚𝙩𝙬𝙚𝙚𝙣(𝙪𝙣𝙗𝙤𝙪𝙣𝙙𝙚𝙙𝙋𝙧𝙚𝙘𝙚𝙙𝙞𝙣𝙜, 𝙘𝙪𝙧𝙧𝙚𝙣𝙩𝙍𝙤𝙬)` to scan all previous records for the most recent non-null value
𝗠𝗲𝗺𝗼𝗿𝘆 𝗢𝗽𝘁𝗶𝗺𝗶𝘇𝗮𝘁𝗶𝗼𝗻 - Maintains original dataset size without creating additional rows for date gaps
𝗚𝗮𝗽 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻 - Optional `𝙡𝙖𝙜()` function to identify and analyze date gaps in the data



𝗪𝗵𝘆 𝗧𝗵𝗶𝘀 𝗔𝗽𝗽𝗿𝗼𝗮𝗰𝗵 𝗪𝗶𝗻𝘀: 
𝗠𝗲𝗺𝗼𝗿𝘆 𝗘𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝗰𝘆 - No additional rows created for missing dates, significantly reducing memory usage
𝗕𝗲𝘁𝘁𝗲𝗿 𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 - Processes fewer rows with window functions operating on smaller datasets
𝗗𝗮𝘁𝗮 𝗜𝗻𝘁𝗲𝗴𝗿𝗶𝘁𝘆 - Preserves distinction between actual and filled data while maintaining original data quality
𝗦𝗰𝗮𝗹𝗮𝗯𝗶𝗹𝗶𝘁𝘆 - Handles large time series datasets without data volume explosion

WINDOW FUNCTION DETAILS:

1. partitionBy("store_id"): 
   - Separates calculations by store
   - Each store gets independent running totals

2. orderBy("order_date"):
   - Ensures chronological processing
   - Critical for forward fill logic

3. rowsBetween(unboundedPreceding, currentRow):
   - Looks at all rows from partition start to current row
   - Enables last() to find most recent non-null value

FORWARD FILL MECHANICS:

For STORE_001 with dates [Jan 1, Jan 2, Jan 4, Jan 5] and orders [15, 22, 18, 25]:
- Jan 1: orders_filled = 15 (actual value)
- Jan 2: orders_filled = 22 (actual value)  
- Jan 4: orders_filled = 18 (actual value, but conceptually Jan 3 would be 22)
- Jan 5: orders_filled = 25 (actual value)

Running totals: [15, 37, 55, 80]

The key insight is that we don't need explicit Jan 3rd row to calculate correct 
running totals. The forward fill logic is applied conceptually during aggregation.

𝗪𝗵𝗲𝗻 𝘁𝗼 𝗖𝗵𝗼𝗼𝘀𝗲 𝗙𝗼𝗿𝘄𝗮𝗿𝗱 𝗙𝗶𝗹𝗹: 
- Running aggregations in time series analysis
- Missing dates should inherit previous values
- Memory efficiency is crucial for large datasets
- Financial calculations like running balances
- Don't need explicit zero values for gaps

𝗪𝗵𝗲𝗻 𝘁𝗼 𝗨𝘀𝗲 𝗗𝗮𝘁𝗲 𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻 𝗜𝗻𝘀𝘁𝗲𝗮𝗱: 
- Need explicit zero values for missing dates
- Reports must display every date
- Distinguish between "no data" vs "zero activity"
- Joining with complete time series datasets
- Performing calculations specifically on date gaps

"""
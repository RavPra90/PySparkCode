from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, floor, rand, expr

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Salting for Skew") \
    .getOrCreate()

# ----------------------------
# 1. Create a dummy DataFrame
# ----------------------------
# Simulate 1M rows: customer "hot" repeats 100K times, others are uniform
hot_rows = [("hot", i) for i in range(100_000)]
other_rows = [("cust_" + str(i % 900), i) for i in range(900_000)]
df = spark.createDataFrame(hot_rows + other_rows, ["customer_id", "value"])

# --------------------------------------
# 2. Detect skew: count per partition
# --------------------------------------
# A simple check: show top customer counts
df.groupBy("customer_id").count() \
  .orderBy(col("count").desc()) \
  .show(3, truncate=False)
# "hot" will dominate, causing one partition to balloon

# ---------------------------------------------------
# 3. Apply salting: evenly distribute the “hot” key
# ---------------------------------------------------
# 3.1. Add a random salt between 0 and (N-1).
#      Choose N such that skewed key is spread across N buckets.
NUM_BUCKETS = 10

salted = (
    df
    .withColumn("salt", floor(rand(seed=42) * NUM_BUCKETS))                  # assign salt
    .withColumn("salted_key", concat(col("customer_id"), lit("_"), col("salt")))  # create composite key
)

# 3.2. Perform aggregation on salted key
agg_salted = (
    salted
    .groupBy("salted_key")
    .count()                          # or sum("value"), avg, etc.
)

# 3.3. Remove salt to get true customer aggregates
final_agg = (
    agg_salted
    .withColumn("customer_id", expr("split(salted_key, '_')[0]"))  # drop salt
    .groupBy("customer_id")
    .sum("count")
    .withColumnRenamed("sum(count)", "total_count")
)

final_agg.show(5, truncate=False)

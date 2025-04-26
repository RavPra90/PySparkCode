from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col, count

spark = SparkSession.builder \
    .appName("DummyWordCountDF") \
    .master("local[*]") \
    .getOrCreate()

# 1. Define dummy lines as a list of tuples (value column)
dummy_data = [
    ("Hello world hello",),
    ("Apache Spark is awesome",),
    ("Hello Spark world",),
    ("Spark makes data processing easy",)
]

# 2. Create DataFrame with schema name = "value"
df = spark.createDataFrame(dummy_data, ["value"])

# 3. Explode and count words
result = (
    df
    .select(explode(split(lower(col("value")), "\\s+")).alias("word"))
    .filter(col("word") != "")
    .groupBy("word")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
)

result.show(truncate=False)

spark.stop()

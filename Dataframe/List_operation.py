from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, broadcast
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ListComparison") \
    .getOrCreate()

#DUMMY DATASET WITH NULLS, DUPLICATES AND EMPTY STRINGS
list_a = ["apple", "banana", "cherry", "apple", None, ""]  # duplicates, null, and empty string
list_b = ["banana", "cherry", "grape", "banana", None, ""]  # duplicates, null, and empty string

# Create DataFrames with explicit schema
schema = StructType([StructField("item", StringType(), True)])
df_a_raw = spark.createDataFrame([(item,) for item in list_a], schema)
df_b_raw = spark.createDataFrame([(item,) for item in list_b], schema)

print("Original DataFrames (with nulls, duplicates, and empty strings):")
df_a_raw.show()
df_b_raw.show()

# CLEAN DATA: Remove nulls, empty strings, and duplicates
df_a = df_a_raw.filter(col("item").isNotNull() & (col("item") != "")).distinct()
df_b = df_b_raw.filter(col("item").isNotNull() & (col("item") != "")).distinct()

print("Cleaned DataFrames (nulls, empty strings, and duplicates removed):")
df_a.show()
df_b.show()

print("Approach 1. Using Joins:")

# Common Items Using INNER JOIN
# TIP: Use broadcast for smaller dataset if size < 200MB
print("COMMON ITEMS (A ∩ B) - Using Inner Join:")
common_items = df_a.join(broadcast(df_b), "item", "inner")
common_items.show()

# Items in A but NOT in B using LEFT ANTI JOIN
# THEORY: Left anti join is specifically designed for "NOT IN" operations
print("ITEMS IN A BUT NOT IN B (A - B) - Using Left Anti Join:")
a_not_in_b = df_a.join(df_b, "item", "left_anti")
a_not_in_b.show()

#  Items in B but NOT in A using LEFT ANTI JOIN
print("ITEMS IN B BUT NOT IN A (B - A) - Using Left Anti Join:")
b_not_in_a = df_b.join(df_a, "item", "left_anti")
b_not_in_a.show()

#Items NOT Common (Symmetric Difference)
# APPROACH: Union of two anti-joins - more efficient than full outer
print("ITEMS NOT COMMON (A ⊕ B) - Using Union of Anti Joins:")
not_common = a_not_in_b.union(b_not_in_a)
not_common.show()

print("Approach 2. Use native operations: intersect(), subtract(), union()")

#ITEMS COMMON TO BOTH LISTS (INTERSECTION)
# Using inner join to find matching items
print("COMMON ITEMS (A ∩ B) - Using INTERSECTION ")
common_items = df_a.select("item") \
    .intersect(df_b.select("item")).show() # Native Spark intersect operation

# ITEMS IN LIST A BUT NOT IN LIST B (A - B)
print("ITEMS IN A BUT NOT IN B (A - B) - Using subtract():")
df_a.subtract(df_b).show()
print("ITEMS IN B BUT NOT IN A (B - A) - Using subtract():")
df_b.subtract(df_a).show()


spark.stop()

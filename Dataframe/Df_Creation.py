# PySpark DataFrame Creation Examples
# This code demonstrates several ways to create a Spark DataFrame,
# with detailed comments for beginners. We'll cover:
#   1. Creating DataFrame from in-memory collections (list of tuples, list of dicts).
#   2. Creating DataFrame from an existing RDD.
#   3. Creating DataFrame by reading from external files (CSV, JSON, Parquet).
#   4. Creating DataFrame by reading from a database (JDBC).

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Initialize SparkSession
# SparkSession is the entry point to programming Spark with DataFrames.
# It allows the creation of DataFrames and access to Spark capabilities.
spark = (SparkSession.builder
    .appName("DataFrameCreationExamples")
    .getOrCreate())

# =============================================================================
# 1. Creating DataFrame from in-memory collections (list of tuples, list of dicts).
# =============================================================================

# Example 1.1: Creating DataFrame from a Python list of tuples.
data_tuples = [
    (1, "Alice", 29),
    (2, "Bob", 31),
    (3, "Charlie", 25)
]
# Each tuple represents a row of data.

# Option A: Let Spark infer the schema and provide column names explicitly.
# We pass the list of tuples and a list of column names to spark.createDataFrame.
# Spark will create a DataFrame with given column names and infer data types from the data.
df_from_tuples = spark.createDataFrame(data_tuples, ["id", "name", "age"])
# Note: Here, schema inference will scan the data to determine column types, which is fine for small data.
# For large datasets, explicitly defining schema can improve performance (see Option B below).

# Check the schema and data of the DataFrame
df_from_tuples.printSchema()  # Shows column names and types
df_from_tuples.show()         # Triggers computation (lazy evaluation) and displays the data

# Option B: Define the schema explicitly using StructType (best practice for large data).
schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("name", StringType(), nullable=True),
    StructField("age", IntegerType(), nullable=True)
])


# Create DataFrame with explicit schema. Spark won't infer types since we provided them.
df_from_tuples_with_schema = spark.createDataFrame(data_tuples, schema)
df_from_tuples_with_schema.printSchema()
df_from_tuples_with_schema.show()

# Example 1.2: Creating DataFrame from a Python list of dictionaries.
data_dicts = [
    {"id": 4, "name": "David", "age": 22},
    {"id": 5, "name": "Eva", "age": 28}
]
# Each dictionary represents a row, with keys as column names.
# When creating a DataFrame from a list of dicts, Spark uses dict keys as column names.
df_from_dicts = spark.createDataFrame(data_dicts)
# Spark infers the schema (column names and types) from the dict keys and values.
df_from_dicts.printSchema()
df_from_dicts.show()

# Note: If dictionaries have missing keys in some rows, Spark will use null for those missing values.

# =============================================================================
# 2. Creating DataFrame from an existing RDD
# =============================================================================

# Create an RDD (Resilient Distributed Dataset) from an in-memory collection.
# RDDs are lower-level distributed collections in Spark.
# We parallelize a Python list into an RDD with 2 partitions.
rdd = spark.sparkContext.parallelize([
    (10, "John"),
    (11, "Jane"),
    (12, "Doe")
], 2)  # '2' sets the number of partitions for parallelism.

# Option A: Convert RDD of tuples to DataFrame by providing column names.
df_from_rdd = spark.createDataFrame(rdd, ["id", "first_name"])
df_from_rdd.printSchema()
df_from_rdd.show()

# Option B: Using Row objects for more complex schema.
# Create an RDD of Row objects (each Row has named fields).
rdd_rows = spark.sparkContext.parallelize([
    Row(id=20, name="Frank"),
    Row(id=21, name="Grace")
])
df_from_rdd_rows = spark.createDataFrame(rdd_rows)
# Column names are taken from the Row field names.
df_from_rdd_rows.printSchema()
df_from_rdd_rows.show()

# Tricky concept - partitions and parallelism:
# The underlying RDD for a DataFrame has partitions. The number of partitions affects parallel processing.
print(f"Number of partitions in DataFrame's RDD: {df_from_rdd.rdd.getNumPartitions()}")
# If needed, repartition the DataFrame to increase or decrease parallelism:
df_repartitioned = df_from_rdd.repartition(3)  # Example: create 3 partitions
print(f"Number of partitions after repartition: {df_repartitioned.rdd.getNumPartitions()}")
# Repartitioning is a transformation (lazy) and will shuffle data when an action is executed.

# =============================================================================
# 3. Creating DataFrame by reading from external files (CSV, JSON, Parquet)
# =============================================================================

# -- Example CSV: --
df_csv = (spark.read
    .option("header", "true")    # Indicate that the first line contains column names
    .option("inferSchema", "true")  # Instruct Spark to infer data types; may be slow for large files
    .option("sep", ",")         # Specify the delimiter (comma is default)
    .csv("E:\PyCharmPythonProjects\PySparkCode\Resources\\read\orders.csv"))  # Path to the CSV directory or file
df_csv.printSchema()
df_csv.show()

# Key parameters for CSV:
# - header=True: first line has column names (default is False).
# - inferSchema=True: automatically detect column data types (expensive on large files).
# - sep: column delimiter (default is comma, e.g. use sep="\\t" for TSV).
# Note: If header=False or missing, Spark assigns default column names (_c0, _c1, ...).

# -- Example JSON: --
# Read JSON file into DataFrame.
df_json = (spark.read
    .option("multiLine", "true")  # Handle multi-line JSON records if needed
    .json("E:\PyCharmPythonProjects\PySparkCode\Resources\\read\multiline-zipcode.json"))  # Path to JSON file or directory
df_json.printSchema()
df_json.show()

# -- Example Parquet: --
# Read Parquet file into DataFrame.
df_parquet = spark.read.parquet("E:\PyCharmPythonProjects\PySparkCode\Resources\\read\weather.parquet")
df_parquet.printSchema()
df_parquet.show()

# Parquet stores schema with the data, so Spark knows column names/types without inference.
# Parquet is efficient (columnar format) and a good choice for large data.

# =============================================================================
# 4. Creating DataFrame by reading from a database (via JDBC)
# =============================================================================
"""
# Example JDBC options (replace with your actual database information).
jdbc_url = "jdbc:postgresql://localhost:5432/mydatabase"
connection_properties = {
    "user": "your_username",
    "password": "your_password",
    "driver": "org.postgresql.Driver"  # JDBC driver for PostgreSQL
}

# Basic JDBC read into DataFrame (lazy, data is not fetched until an action is called).
df_db = (spark.read
    .jdbc(url=jdbc_url, table="public.my_table", properties=connection_properties))
df_db.printSchema()
df_db.show(5)  # Shows first 5 rows; triggers SQL query to the database

# Best practice for large tables: partitioning
df_db_partitioned = (spark.read
    .format("jdbc")
    .option("url", jdbc_url)
    .option("dbtable", "public.my_large_table")
    .option("user", connection_properties["user"])
    .option("password", connection_properties["password"])
    .option("driver", connection_properties["driver"])
    .option("partitionColumn", "id")   # Column to split data for parallel reads
    .option("lowerBound", "1")         # Lower bound of values in the partition column
    .option("upperBound", "100000")    # Upper bound of values in the partition column
    .option("numPartitions", "4")      # Number of parallel partitions (connections)
    .load())
# Spark will generate 4 parallel queries, splitting the 'id' range into chunks.
df_db_partitioned.printSchema()
df_db_partitioned.show(5)

# Note: Replace URL, table names, and credentials with your database details.
# Ensure the JDBC driver (e.g., for PostgreSQL) is on the Spark classpath.
"""
# Finally, stop the SparkSession when done to free resources.
spark.stop()

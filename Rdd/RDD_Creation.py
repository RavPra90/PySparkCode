from pyspark import SparkContext, SparkConf


# 1. Initialize SparkContext
# --------------------------
# A SparkContext represents the connection to a Spark cluster, and is your entry point to use Spark.
# Best practice: configure SparkConf separately for clarity and easier tuning.
conf = SparkConf().setAppName("RDDCreationExamples").setMaster("local[*]")
sc = SparkContext(conf=conf)
# Note: "local[*]" runs Spark locally with as many worker threads as logical cores on your machine.

# ---------------------------------------------------------------------------------------------------
# Example 1: Creating an RDD from an in-memory Python collection using `parallelize`
# ---------------------------------------------------------------------------------------------------
# This is the simplest way to create an RDD when you already have data in your Python program.
# parallelize distributes the data across the cluster in partitions (default is 2 partitions).

numbers = [1, 2, 3, 4, 5]
rdd_from_list = sc.parallelize(numbers)

# Tricky concept: By default, Spark decides the number of partitions, but you can control it:
# sc.parallelize(numbers, numSlices=4)
# Best practice: Choose partitions based on data size and cluster resources.

print("Example 1 RDD Contents:", rdd_from_list.collect())

# ---------------------------------------------------------------------------------------------------
# Example 2: Creating an RDD by reading a text file using `textFile`
# ---------------------------------------------------------------------------------------------------
# When working with external data (e.g., text files on HDFS, S3, or local FS), use textFile.
# Each line in the file becomes an element in the resulting RDD.

path = "E:\PyCharmPythonProjects\PySparkCode\Resources\RDD"  # Replace with your file path
text_rdd = sc.textFile(path)

# Important: textFile is lazy. No data is read until an action (e.g., count(), collect()) is called.
# Action below triggers reading and counting lines.
print("Number of lines in text file:", text_rdd.count())

# ---------------------------------------------------------------------------------------------------
# Example 3: Creating an RDD from a directory of files using `wholeTextFiles`
# ---------------------------------------------------------------------------------------------------
# wholeTextFiles reads a directory and returns an RDD of (filename, content) pairs.
# Useful when you need file-level context.

dir_path = "E:\PyCharmPythonProjects\PySparkCode\Resources"
files_rdd = sc.wholeTextFiles(dir_path)

# Each record is a tuple (filename, fileContent)
#for filename, content in files_rdd.take(3):
    #print(f"File: {filename}\nContent snippet: {content[:100]}\n")

# ---------------------------------------------------------------------------------------------------
# Example 4: Creating an Empty RDD
# ---------------------------------------------------------------------------------------------------
# Sometimes, you may need an empty RDD as a placeholder or initial accumulator.
empty_rdd = sc.emptyRDD()
print("Is the RDD empty?", empty_rdd.isEmpty())  # returns True

# ---------------------------------------------------------------------------------------------------
# Example 5: Converting a DataFrame to an RDD
# ---------------------------------------------------------------------------------------------------
# While Spark DataFrames offer optimizations, you may need raw RDDs for low-level transformations.
from pyspark.sql import SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Create a DataFrame
df = spark.createDataFrame([("Alice", 34), ("Bob", 45)], ["name", "age"])

# Convert DataFrame rows to RDD of Row objects
df_rdd = df.rdd
print("DF -> RDD Contents:", df_rdd.collect())

# Tricky concept: df.rdd gives an RDD of Row objects. You often need to extract fields:
ages = df_rdd.map(lambda row: row['age']).collect()
print("Extracted ages:", ages)

# ---------------------------------------------------------------------------------------------------
# Example 6: Creating an RDD from Sequence Files (Hadoop) using `sequenceFile`
# ---------------------------------------------------------------------------------------------------
# For advanced Hadoop integrations, you can read binary sequence files:
# sequenceFile(path, keyClass, valueClass) returns an RDD of (key, value).
# You'll need the Hadoop classes available in your Spark cluster.
#
# seq_rdd = sc.sequenceFile("data/seqfiles", "org.apache.hadoop.io.Text", "org.apache.hadoop.io.IntWritable")
# print(seq_rdd.take(5))

"""
# ---------------------------------------------------------------------------------------------------
# Best Practices and Tips:
# - Avoid calling collect() on large RDDs; use take(), count(), or saveAsTextFile() instead.
# - Remember RDDs are immutable: each transformation creates a new RDD.
# - Use persist()/cache() for RDDs that will be reused across multiple actions to avoid re-computation.
# - Be mindful of closures: avoid referencing large objects from driver inside map() to prevent serialization overhead.
# - Partitioning: Use repartition() or coalesce() when you need to change the number of partitions for performance tuning.

"""

# Clean up SparkContext
sc.stop()

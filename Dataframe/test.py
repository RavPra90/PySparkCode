from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Initialize SparkSession
# SparkSession is the entry point to programming Spark with DataFrames.
# It allows the creation of DataFrames and access to Spark capabilities.
spark = (SparkSession.builder
    .appName("DataFrameCreationExamples")
    .getOrCreate())

sample_data = [("Sam", 23), ("Anna", 27)]
df_sample = spark.createDataFrame(sample_data, ["name", "age"])
# Write DataFrame to a CSV (directory with part files). We use coalesce(1) for a single output file for simplicity.
df_sample.coalesce(1).write.csv("E:\\PyCharmPythonProjects\PySparkCode\Resources\write\sample_data.csv", header=True, mode="overwrite")
#
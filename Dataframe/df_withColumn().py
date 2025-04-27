# Import SparkSession and functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create a Spark session
spark = SparkSession.builder \
    .appName("FilterWhereTutorial") \
    .getOrCreate()

# Read the orders.csv data (header row, infer data types)
df = spark.read.csv("E:\PyCharmPythonProjects\PySparkCode\Resources\\read\orders.csv", header=True, inferSchema=True)
df.show(5)

#Adding a New Column
#======================================================================================================================
"""
To add a new column, use withColumn(colName, expression). 
Here we create a column order_year by extracting the year from the order_date string. 
We first convert order_date to a date and then apply the year() function.
"""
# Add a new column 'order_year' by extracting the year from 'order_date'
df = df.withColumn(
    "order_year",
    year(to_date(col("order_date"), "yyyy-MM-dd HH:mm:ss.S"))
)
df.show(5)
"""
Explanation: This uses to_date() to parse the timestamp string (format "yyyy-MM-dd HH:mm:ss.S") and then year() 
to get the year. The new column order_year is appended to the DataFrame. 
Since withColumn returns a new DataFrame, we reassign it to df. 
If an order date is missing (null), order_year will be null as well unless we handle it (we’ll see null handling next).
"""

#Modifying (Replacing) an Existing Column
#====================================================================================================================
"""
If you call withColumn() with the name of an existing column, it replaces that column
"""
df.withColumn("order_status", lower("order_status"))
df.show(5)
"""
Using lower(col("order_status")) converts text to lowercase. Because we passed the same column name "order_status" as the first argument, 
the original column is replaced with this new lowercase version . If you wanted to keep the original, use a new name:
"""
df.withColumn("new_status", lower(col("order_status")))

#Handling Null or Missing Data
#====================================================================================================================
# Create 'order_year' again, assigning -1 where 'order_date' is null
df = df.withColumn(
    "order_year",
    when(col("order_date").isNull(), lit(-1).cast(IntegerType()))
    .otherwise(year(to_date(col("order_date"), "yyyy-MM-dd HH:mm:ss.S")))
)
df.show(5)
"""
Explanation: We use when(condition, value).otherwise(other_value) to handle the null case. 
Here, if order_date is null, order_year is set to -1 (cast to Integer). Otherwise, we compute the year normally. 
The when function returns a Column expression, similar to SQL’s CASE statement.
If otherwise() were omitted and no condition matched, the result would be null.
"""

#Conditional Columns with when / otherwise
#======================================================================================================================
"""
when/otherwise is also useful for creating conditional flags or categories. For instance, flag completed orders:
"""
# Add a binary flag 'is_complete' (1 if status is COMPLETE, else 0)
df = df.withColumn(
    "is_complete",
    when(col("order_status") == "COMPLETE", 1).otherwise(0)
)

# Add a label based on status
df = df.withColumn(
    "status_label",
    when(col("order_status") == "COMPLETE", "Complete Order")
    .otherwise("Other Order")
)
df.show(5)

# Chaining Multiple withColumn Calls
#===================================================================================================================
"""
You can chain several withColumn() calls one after another.
However, note that each withColumn adds an internal projection to the query plan. 
Spark’s documentation warns that chaining many transformations can lead to large execution plans and performance issues
"""
# Example: chain two withColumn calls
df = df.withColumn("status_lower", lower(col("order_status"))) \
       .withColumn("customer_type",
                   when(col("order_customer_id") > 10000, "VIP").otherwise("Regular"))
df.show(5)

#For a more dynamic example, you can programmatically add multiple columns in a loop:
# Dynamically add indicator columns for each status type
statuses = ["COMPLETE", "CLOSED", "PENDING_PAYMENT"]
for status in statuses:
    df = df.withColumn(f"is_{status.lower()}", (col("order_status") == status).cast("int"))
df.show(5)

"""
Performance and Best Practices
================================================================================================================================
Use explicit schema: Providing a schema when reading (rather than inferSchema) avoids the extra pass over data
Minimize transformations: Each withColumn() introduces a projection. 
                          Chaining many calls (especially inside loops) can bloat the execution plan and slow down queries
Combine operations when possible: If you need to add or modify many columns at once, consider using a single select() 
                            with multiple expressions (Spark can optimize a single project step better than many sequential withColumns)
Careful naming: Remember that withColumn(name, ...) replaces an existing column if name already exists .
                To avoid confusion, use new column names or explicitly drop/rename old columns before overwriting.
Handle nulls explicitly: Decide on null handling (fill, default value, or filter out) before adding derived columns, 
                        to avoid unexpected nulls in calculations.
Cache intermediate results: If a DataFrame is reused for multiple transformations or actions, 
                            use .cache() to store it in memory and avoid recomputation.
Avoid UDFs when possible: Use built-in functions (upper, year, when, etc.) instead of Python UDFs for better performance.
"""
# Import SparkSession and functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder \
    .appName("FilterWhereTutorial") \
    .getOrCreate()

# Read the orders.csv data (header row, infer data types)
df = spark.read.csv("E:\PyCharmPythonProjects\PySparkCode\Resources\\read\orders.csv", header=True, inferSchema=True)
df.show(5)

"""
Use whichever reads better for you .The Spark API doc explicitly notes “where() is an alias for filter()
In practice, no performance difference exists between them
"""

#Here df.order_status == "COMPLETE" creates a Boolean column for the condition, and filter() returns rows
# where it’s True. The result includes only orders whose order_status is exactly "COMPLETE".
complete_orders= df.filter(df.order_status == "COMPLETE")
complete_orders.show(5)

#using where ()
complete_orders= df.where(df.order_status == "COMPLETE")
complete_orders.show(5)

#Filtering with SQL-like String Expressions

# Filter using a SQL expression string (notice single quotes around text)
pending_orders = df.filter("order_status = 'PENDING'")
pending_orders.show(5)

#Here we chained two filters: one for status and another for customer_id.
# Chaining filters is equivalent to using & to combine them in one call.
# For clarity, you could also write it as one expression:
filtered = df.filter(
    (col("order_status") == "COMPLETE") |
    (col("order_status") == "CLOSED")
).filter(col("order_customer_id") > 10000)

#Or combine both into a single filter with &:
filtered3 = df.filter(
    ((col("order_status") == "COMPLETE") | (col("order_status") == "CLOSED")) &
    (col("order_customer_id") > 10000)
)
filtered3.show(5)

"""
Remember to use parentheses when mixing & and |, because & and | have higher precedence in Python and work on Spark Column Objects
(Also note: don’t use Python’s and/or – Spark requires the bitwise &/| for column logic).
 For SQL strings, you could similarly write "order_status = 'COMPLETE' OR order_status = 'CLOSED'".
"""

#Filtering Null and Missing Values

"""
Real-world data often has missing (null) values. In Spark, you cannot filter for None using == None or = null directly.
 Instead, use the .isNull() or .isNotNull() methods on a Column
"""
# Show rows where order_customer_id is null
null_id_orders = df.filter(col("order_customer_id").isNull())
null_id_orders.show()

#SQL SYNTAX
df.filter("order_customer_id IS NULL").show()

#String Pattern Matching
#----------------------------
"""
Besides exact equality, you can filter strings using methods like .contains(), .like(), 
or regular expressions (e.g. .rlike()). For example, to find all orders with status containing "PENDING":
"""
# Filter where order_status contains "PENDING"
pending_like = df.filter(col("order_status").contains("PENDING"))
pending_like.show(5)

# Or using SQL-style LIKE (case-sensitive)
pending_like2 = df.filter(col("order_status").like("%PENDING%"))
pending_like2.show(5)
"""
These keep rows with "PENDING" anywhere in the status. Be mindful of case sensitivity: 
Spark string comparisons are case-sensitive, so "complete" would not match "COMPLETE" 
unless you transform cases (e.g. using lower(col("order_status")) == "complete").
"""


"""
Best Practices and Tips
Chain early: Apply filters as early as possible in your pipeline to reduce data size and computation
For large datasets, filtering at the source (or immediately after reading) can save a lot of work.
Maintain readability: When conditions get complex, use parentheses and/or multiple .filter() calls to keep things clear. 
                    For example, separating conditions across lines or using col() can improve legibility.
Use column expressions: Passing a Column (e.g. col("status") == "X") helps IDEs catch typos and supports more functions.
                        String queries ("status = 'X'") are quick but hide errors until runtime.
Beware partitions: A Spark filter does not automatically shrink partitions. If you filter a huge dataset down
                    to a tiny result, you may still have many empty partitions, which can hurt performance  
                    
In production, you might follow a big filter with repartition() or coalesce() to reduce partitions if the result is much smaller

Handle nulls explicitly: Decide how to treat null or missing values. Sometimes filtering them out (using .isNotNull() or dropna) is important for correctness.
Test filter logic: Use .show() or .count() to verify filters work as intended, especially with string quoting and date formats. For example, ensure you use the correct datetime format if filtering dates stored as strings.

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# PySpark Joins and Join Optimization - Comprehensive Guide

This guide covers PySpark joins from the basics to advanced optimization techniques,
with a special focus on optimizing joins between large fact tables and small dimension tables.

## Table of Contents:
1. Basic Join Types
2. Join Conditions and Syntax
3. Join Performance Considerations
4. Broadcast Joins
5. Sort-Merge Joins
6. Shuffle Hash Joins
7. Optimizing Star Schema Joins (Fact-Dimension)
8. Advanced Optimization Techniques
9. Handling Skew
10. Monitoring and Debugging Joins
"""
from pyspark.sql.connect.session import SparkSession
from pyspark.sql.functions import col, broadcast, expr, count, sum, avg, max, lit
import pyspark.sql.functions as F
from pyspark.sql import SparkSession


# Create a SparkSession
spark = SparkSession.builder\
    .appName("PySpark Joins and Optimization Guide")\
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
    .getOrCreate()

# Set log level to minimize console output
spark.sparkContext.setLogLevel("WARN")

#############################################################
# SECTION 1: BASIC JOIN TYPES
#############################################################

"""
PySpark supports various join types that correspond to SQL join types:

1. Inner Join: Returns only matching rows between tables
2. Left (Outer) Join: Returns all rows from the left table and matching rows from the right
3. Right (Outer) Join: Returns all rows from the right table and matching rows from the left
4. Full (Outer) Join: Returns all rows when there's a match in either table
5. Cross Join: Returns the Cartesian product of both tables (every combination)
6. Semi Join: Returns rows from the left table that have a match in the right table
7. Anti Join: Returns rows from the left table that DON'T have a match in the right table

Let's demonstrate all these join types with some sample data.
"""

# Create sample dataframes to represent tables
# Employee table
employee_data = [
    (1, "John", "Engineering"),
    (2, "Jane", "Marketing"),
    (3, "Bob", "Sales"),
    (4, "Alice", "Engineering"),
    (5, "David", "Finance")
]
employee_df = spark.createDataFrame(employee_data, ["emp_id", "name", "department"])

# Department table with some departments missing to demonstrate different join behaviors
department_data = [
    ("Engineering", "Building A"),
    ("Marketing", "Building B"),
    ("HR", "Building C"),
    ("Finance", "Building A")
]
department_df = spark.createDataFrame(department_data, ["department", "location"])

# Cache these small dataframes since we'll reuse them multiple times
employee_df.cache()
department_df.cache()

print("Employee DataFrame:")
employee_df.show()

print("Department DataFrame:")
department_df.show()

# 1. INNER JOIN
# Returns only the rows that have matching values in both dataframes
print("INNER JOIN Example:")
inner_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "inner"  # Join type (could also use just "join" without specifying type as inner is default)
)
inner_join_df.show()
# Notice: Bob in Sales is missing because there's no matching department

# 2. LEFT OUTER JOIN
# Returns all rows from the left dataframe and matching rows from the right
print("LEFT JOIN Example:")
left_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "left"  # or "left_outer"
)
left_join_df.show()
# Notice: Bob appears with NULL for location since Sales has no match

# 3. RIGHT OUTER JOIN
# Returns all rows from the right dataframe and matching rows from the left
print("RIGHT JOIN Example:")
right_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "right"  # or "right_outer"
)
right_join_df.show()
# Notice: HR appears with NULL for employee fields since no employee is in HR

# 4. FULL OUTER JOIN
# Returns all rows when there's a match in either left or right dataframe
print("FULL OUTER JOIN Example:")
full_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "full"  # or "full_outer" or "outer"
)
full_join_df.show()
# Notice: Both Bob (Sales) and HR department appear with NULLs where no matches exist

# 5. CROSS JOIN
# Returns the Cartesian product - every combination of rows
print("CROSS JOIN Example (limited to 10 rows):")
cross_join_df = employee_df.crossJoin(department_df)
cross_join_df.show(10)  # Limiting output as cross joins can produce a lot of rows
# Notice: Each employee is paired with every department regardless of their actual department

# 6. LEFT SEMI JOIN
# Similar to EXISTS in SQL -  Returns only rows from the left table for which there is at least one matching row in the right table.
print("LEFT SEMI JOIN Example:")
semi_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "left_semi"
)
semi_join_df.show()
# Notice: Only employees whose departments exist in department_df, but no columns from department_df

# 7. LEFT ANTI JOIN
# Opposite of semi join - returns rows from left that DON'T have matches in right
print("LEFT ANTI JOIN Example:")
anti_join_df = employee_df.join(
    department_df,
    employee_df.department == department_df.department,
    "left_anti"
)
anti_join_df.show()
# Notice: Only Bob appears because Sales doesn't exist in department_df

#############################################################
# SECTION 2: JOIN CONDITIONS AND SYNTAX
#############################################################

"""
PySpark offers multiple ways to specify join conditions:

1. Using the join method with condition
2. Using the join method with columns as strings
3. Multiple join conditions
4. Joining on different column names
5. Complex join conditions

Let's look at different join syntaxes:
"""

# 1. Basic join condition using column objects
join_using_cols = employee_df.join(
    department_df,
    employee_df.department == department_df.department
)

# 2. Join using column names as strings
join_using_strings = employee_df.join(
    department_df,
    "department" # When column names are the same in both dataframes
)

# 3. Multiple join conditions (rarely needed for dimension-fact tables but included for completeness)
# Let's add a second matching field to both dataframes
employee_with_code_df = employee_df.withColumn("dept_code",
                                               F.when(col("department") == "Engineering", "ENG")
                                               .when(col("department") == "Marketing", "MKT")
                                               .when(col("department") == "Sales", "SLS")
                                               .when(col("department") == "Finance", "FIN")
                                               .otherwise(None)
                                               )

department_with_code_df = department_df.withColumn("dept_code",
                                                   F.when(col("department") == "Engineering", "ENG")
                                                   .when(col("department") == "Marketing", "MKT")
                                                   .when(col("department") == "HR", "HRS")
                                                   .when(col("department") == "Finance", "FIN")
                                                   .otherwise(None)
                                                   )

# Join on multiple conditions
multi_condition_join = employee_with_code_df.join(
    department_with_code_df,
    (employee_with_code_df.department == department_with_code_df.department) &
    (employee_with_code_df.dept_code == department_with_code_df.dept_code)
)

# 4. Join when columns have different names
employee_renamed_df = employee_df.withColumnRenamed("department", "emp_department")
different_name_join = employee_renamed_df.join(
    department_df,
    employee_renamed_df.emp_department == department_df.department
)

# 5. Using more complex expressions in join conditions (using expr)
complex_join = employee_df.join(
    department_df,
    expr("employee_df.department = department_df.department AND " +
         "department_df.location LIKE '%Building A%'")
)

print("Complex join example (with location filter):")
complex_join.show()

# 6. Using joinWith to keep the original dataframes' structures
# Returns a dataset of pairs (tuple2) containing matching rows
typed_join = employee_df.joinWith(
    department_df,
    employee_df.department == department_df.department,
    "inner"
)

#############################################################
# SECTION 3: JOIN PERFORMANCE CONSIDERATIONS
#############################################################

"""
Understanding PySpark's join strategies is crucial for performance optimization.
The three main join strategies are:

1. Broadcast Join (aka Map-side Join):
   - One small table is broadcast to all executors
   - Best for joining a large table with a small table
   - No shuffling of the large table

2. Sort-Merge Join:
   - Default for large tables
   - Both tables are shuffled by join key, sorted, and then merged
   - Good when both tables are large

3. Shuffle Hash Join:
   - Builds a hash table on the smaller side after shuffling
   - Less common in modern Spark versions due to AQE

Let's look at factors that affect join performance:
"""

# Create larger dataframes to demonstrate performance considerations
# Let's simulate a sales fact table and product dimension table

import random
from datetime import datetime, timedelta


# Helper function to generate realistic data
def generate_sales_data(num_records, num_products, num_customers, num_stores):
    data = []
    start_date = datetime(2020, 1, 1)

    for i in range(num_records):
        product_id = random.randint(1, num_products)
        customer_id = random.randint(1, num_customers)
        store_id = random.randint(1, num_stores)
        quantity = random.randint(1, 10)
        price = round(random.uniform(5.0, 500.0), 2)
        sale_date = start_date + timedelta(days=random.randint(0, 730))

        data.append((i, product_id, customer_id, store_id, quantity, price, sale_date))

    return data


# Generate product dimension data
def generate_product_data(num_products):
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books", "Food", "Toys"]
    suppliers = ["SupplierA", "SupplierB", "SupplierC", "SupplierD", "SupplierE"]

    data = []
    for i in range(1, num_products + 1):
        name = f"Product-{i}"
        category = random.choice(categories)
        supplier = random.choice(suppliers)
        weight = round(random.uniform(0.1, 25.0), 2)

        data.append((i, name, category, supplier, weight))

    return data


# Generate a small sales dataset for this example
# In real scenarios, you might have millions of records
sales_data = generate_sales_data(10000, 100, 500, 50)
product_data = generate_product_data(100)

# Create dataframes
sales_schema = ["sale_id", "product_id", "customer_id", "store_id", "quantity", "price", "sale_date"]
sales_df = spark.createDataFrame(sales_data, schema=sales_schema)

product_schema = ["product_id", "product_name", "category", "supplier", "weight"]
product_df = spark.createDataFrame(product_data, schema=product_schema)

# For demonstration, cache these dataframes
sales_df.cache()
product_df.cache()

sales_df.createOrReplaceTempView("sales")
product_df.createOrReplaceTempView("products")

# Check the sizes of our dataframes
print("Sales DF Count:", sales_df.count())
print("Product DF Count:", product_df.count())

# IMPORTANT: In a real scenario with a 500GB fact table, you would see something like:
# Sales DF Count: 2,000,000,000+ rows
# Product DF Count: 1,000 rows

# Let's look at what affects join performance:

# 1. Data skew
# Some values in the join key may appear much more frequently than others
# Let's check the distribution of products in sales
print("Product distribution in sales:")
sales_df.groupBy("product_id").count().orderBy("count", ascending=False).show(5)

# 2. Handling duplicate column names
# After a join, both dataframes may have columns with the same name
# This can be confusing and lead to errors

# Bad practice - keeping duplicate columns without renaming
bad_join = sales_df.join(product_df, "product_id")

# Now you have two 'product_id' columns which can be confusing
print("Columns in the joined dataframe (with duplicates):")
print(bad_join.columns)

# Better practice - select specific columns or rename them
good_join = sales_df.join(
    product_df,
    sales_df.product_id == product_df.product_id
).select(
    sales_df.sale_id,
    sales_df.product_id,
    product_df.product_name,
    sales_df.quantity,
    sales_df.price,
    (sales_df.quantity * sales_df.price).alias("total_amount")
)

print("Good join with selected columns:")
good_join.show(5)

#############################################################
# SECTION 4: BROADCAST JOINS
#############################################################

"""
Broadcast joins are a key optimization technique when joining a large fact table
with small dimension tables - exactly the scenario described in our problem.

When using broadcast joins:
1. The small table is sent to all executor nodes
2. No shuffling of the large table is needed
3. The join happens locally on each partition of the large table

This significantly reduces network traffic and improves performance.
"""

# Spark automatically broadcasts tables smaller than spark.sql.autoBroadcastJoinThreshold
# The default is 10MB but you can adjust it

# 1. Explicit broadcasting of small tables
broadcast_join = sales_df.join(
    broadcast(product_df),
    sales_df.product_id == product_df.product_id
)

print("Broadcast join example:")
broadcast_join.select(
    "sale_id", "product_id", "product_name", "quantity", "price"
).show(5)

# 2. Checking if broadcast join was used
# You can see the execution plan to confirm
print("Execution plan for the broadcast join:")
broadcast_join.explain()
# Look for "BroadcastHashJoin" in the output

# 3. When to use broadcast joins:
# - One table is small enough to fit in memory
# - One table is significantly smaller than the other
# - The join key is not highly skewed

# 4. Setting broadcast threshold globally
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100mb")
# This increases the size threshold for automatic broadcasting

# 5. Disabling automatic broadcasting (forces sort-merge join)
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# 6. Using SQL hint for broadcasting
# SQL way with broadcast hint
broadcast_sql = spark.sql("""
    SELECT /*+ BROADCAST(p) */ s.sale_id, p.product_name, s.quantity, s.price
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
""")
print("SQL with broadcast hint:")
broadcast_sql.show(5)

#############################################################
# SECTION 5: SORT-MERGE JOINS
#############################################################

"""
Sort-Merge Joins are Spark's default strategy for joining large tables
when broadcast joins aren't possible.

Process:
1. Shuffle both dataframes by the join key
2. Sort both dataframes by the join key
3. Merge the sorted data together

Best when:
- Both tables are large
- The join key is already sorted or has low cardinality
- Broadcasting won't work due to memory constraints
"""

# Force a sort-merge join by disabling broadcast temporarily
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# Now perform a join that will use sort-merge
sort_merge_join = sales_df.join(
    product_df,
    sales_df.product_id == product_df.product_id
)

print("Sort-Merge Join execution plan:")
sort_merge_join.explain()
# Look for "SortMergeJoin" in the output

# Reset the broadcast threshold to default
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")

# Tips for optimizing sort-merge joins:
# 1. Ensure join keys have a good distribution (not skewed)
# 2. Pre-partition large tables by join key
# 3. Use the same partitioning scheme for tables frequently joined together

# Example with repartitioning by join key
sales_repartitioned = sales_df.repartition(10, "product_id")  # 10 partitions by product_id
product_repartitioned = product_df.repartition(10, "product_id")

optimized_join = sales_repartitioned.join(
    product_repartitioned,
    sales_repartitioned.product_id == product_repartitioned.product_id
)

print("Join with repartitioned tables:")
optimized_join.select("sale_id", "product_name", "price").show(5)

#############################################################
# SECTION 6: SHUFFLE HASH JOINS
#############################################################

"""
Shuffle Hash Join is a strategy where:

1. Both tables are shuffled by join key (like sort-merge)
2. A hash table is built on the smaller table's partitions
3. The larger table's partitions probe this hash table

This is typically faster than sort-merge when:
- The tables don't need to be completely sorted
- One table is smaller but still too large to broadcast

Note: In newer Spark versions with Adaptive Query Execution (AQE),
Spark might dynamically switch to this strategy.
"""

# Shuffle Hash Joins are not explicitly specified but can be selected by the optimizer
# Enable Adaptive Query Execution if not already enabled
spark.conf.set("spark.sql.adaptive.enabled", "true")

# Create a medium-sized dataframe that's too large to broadcast but smaller than sales
medium_df = spark.range(0, 100000).selectExpr(
    "id % 100 as product_id",
    "id as medium_id",
    "concat('Item', cast(id as string)) as item_name"
)

# Join that might use shuffle hash join (depending on statistics and AQE)
hash_join = sales_df.join(
    medium_df,
    sales_df.product_id == medium_df.product_id
)

print("Execution plan that might use Shuffle Hash Join:")
hash_join.explain()

# With AQE, Spark can dynamically convert sort-merge joins to broadcast or shuffle hash joins
# when it determines at runtime that a table partition is small enough

#############################################################
# SECTION 7: OPTIMIZING STAR SCHEMA JOINS (FACT-DIMENSION)
#############################################################

"""
Now let's focus on the specific scenario mentioned:
- A 500 GB sales fact table
- Several small dimension tables (a few MB each)

This is a classic star schema data warehouse pattern, and here's how to optimize it:
"""

# Let's create a more realistic star schema example with:
# - Sales fact table (large)
# - Product dimension (small)
# - Customer dimension (small)
# - Store dimension (small)
# - Date dimension (small)

# Creating more dimension tables (already have product_df)
customer_data = [(i, f"Customer-{i}", random.choice(["High", "Medium", "Low"]),
                  random.choice(["US", "UK", "CA", "AU", "DE"]))
                 for i in range(1, 501)]
customer_df = spark.createDataFrame(customer_data, ["customer_id", "customer_name", "segment", "country"])

store_data = [(i, f"Store-{i}", random.choice(["Urban", "Suburban", "Rural"]),
               random.choice(["North", "South", "East", "West"]))
              for i in range(1, 51)]
store_df = spark.createDataFrame(store_data, ["store_id", "store_name", "type", "region"])

# In a real 500GB dataset, you'd have much more data:
# Facts: ~2 billion rows
# Dimensions: 1K-1M rows each

# Cache these small dimension tables
customer_df.cache()
store_df.cache()


# product_df is already cached

# STRATEGY 1: Broadcast all dimension tables
# This is the most effective for star schema queries

def star_schema_query_with_broadcast():
    """
    Demonstrates the optimal way to join a large fact table with multiple dimension tables
    using broadcast joins.
    """
    return sales_df.join(
        broadcast(product_df),
        sales_df.product_id == product_df.product_id
    ).join(
        broadcast(customer_df),
        sales_df.customer_id == customer_df.customer_id
    ).join(
        broadcast(store_df),
        sales_df.store_id == store_df.store_id
    ).select(
        sales_df.sale_id,
        product_df.product_name,
        customer_df.customer_name,
        store_df.store_name,
        sales_df.quantity,
        sales_df.price,
        (sales_df.quantity * sales_df.price).alias("total_amount")
    )


star_result = star_schema_query_with_broadcast()
print("Star schema join result (optimized with broadcasts):")
star_result.show(5)

print("Star schema join execution plan:")
star_result.explain()

# STRATEGY 2: Using SQL for star schema queries
# SQL can be more readable for complex star schema queries

spark.sql("""
    SELECT 
        s.sale_id,
        p.product_name,
        c.customer_name,
        st.store_name,
        s.quantity,
        s.price,
        s.quantity * s.price as total_amount
    FROM 
        sales s
        JOIN products p ON s.product_id = p.product_id
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN stores st ON s.store_id = st.store_id
""").createOrReplaceTempView("detailed_sales")

# Now we can query the detailed view
aggregated_result = spark.sql("""
    SELECT 
        p.category,
        st.region,
        SUM(s.quantity * s.price) as total_sales
    FROM 
        sales s
        JOIN products p ON s.product_id = p.product_id
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN stores st ON s.store_id = st.store_id
    GROUP BY 
        p.category, 
        st.region
    ORDER BY 
        total_sales DESC
""")

print("Aggregated results by category and region:")
aggregated_result.show()


# STRATEGY 3: Choosing the right join order
# When joining multiple tables, join order matters!
# Rule of thumb: Start with the most selective joins first

# Better join order (assuming filters make this more selective)
def selective_join_order():
    return sales_df.filter(col("price") > 100) \
        .join(broadcast(product_df), sales_df.product_id == product_df.product_id) \
        .filter(col("category") == "Electronics") \
        .join(broadcast(customer_df), sales_df.customer_id == customer_df.customer_id)


# The query optimizer will try to reorder joins, but helping it with selective filters is good practice

#############################################################
# SECTION 8: ADVANCED OPTIMIZATION TECHNIQUES
#############################################################

"""
Beyond basic broadcasting, there are several advanced techniques
to optimize joins, especially for large-scale data:
"""

# 1. BUCKETING
# Pre-organizing data into buckets by join key
# This reduces shuffle during joins

# Create bucketed tables
spark.sql("CREATE DATABASE IF NOT EXISTS optimization_demo")
spark.sql("USE optimization_demo")

# Create a bucketed sales table with 10 buckets on product_id
spark.sql("""
    CREATE TABLE IF NOT EXISTS bucketed_sales (
        sale_id INT,
        product_id INT,
        customer_id INT,
        store_id INT,
        quantity INT,
        price DOUBLE,
        sale_date TIMESTAMP
    )
    CLUSTERED BY (product_id) INTO 10 BUCKETS
""")

# Similarly, create a bucketed products table
spark.sql("""
    CREATE TABLE IF NOT EXISTS bucketed_products (
        product_id INT,
        product_name STRING,
        category STRING,
        supplier STRING,
        weight DOUBLE
    )
    CLUSTERED BY (product_id) INTO 10 BUCKETS
""")

# In production, you would insert data into these tables
# For demonstration, we'll just explain the concept:
print("Bucketing explanation:")
"""
When tables are bucketed on the same key with the same number of buckets,
Spark can avoid shuffling during joins. Each bucket of the fact table
only needs to be joined with the corresponding bucket of the dimension table.
This is like having pre-partitioned data.
"""

# 2. BLOOM FILTER JOINS
# Using probabilistic data structures to filter out non-matching rows early
# Available in newer Spark versions (3.0+)

spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")

# When this is enabled, Spark can use Bloom filters to optimize join performance
# by filtering out non-matching rows before the actual join happens

# 3. ADAPTIVE QUERY EXECUTION (AQE)
# Let Spark dynamically optimize joins at runtime

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# With AQE, Spark can:
# - Dynamically coalesce shuffle partitions
# - Convert sort-merge joins to broadcast joins when possible
# - Handle skewed data during joins

# 4. CACHING DIMENSION TABLES
# We've already seen this, but it's worth emphasizing

# 5. PREDICATE PUSHDOWN
# Filter data before the join to reduce data volume

filtered_join = sales_df.filter(col("price") > 200) \
    .join(
    product_df.filter(col("category") == "Electronics"),
    on="product_id"
)

print("Predicate pushdown example:")
filtered_join.select("sale_id", "product_name", "price").show(5)

# 6. COLUMN PRUNING
# Only select the columns you need before joining

pruned_join = sales_df.select("sale_id", "product_id", "price") \
    .join(
    product_df.select("product_id", "product_name"),
    on="product_id"
)

print("Column pruning example:")
pruned_join.show(5)

# 7. PARTITIONING
# When writing data to disk, partition by commonly used join keys

# Example of writing partitioned data (simulated)
# sales_df.write.partitionBy("product_id").saveAsTable("partitioned_sales")

# Then reading becomes more efficient as you can skip irrelevant partitions
# filtered_sales = spark.read.table("partitioned_sales").filter(col("product_id") == 42)

#############################################################
# SECTION 9: HANDLING SKEW
#############################################################

"""
Data skew is one of the biggest challenges for join performance.
It occurs when some values of the join key appear much more frequently than others,
causing uneven distribution of work across executors.
"""

# 1. Identify skew by analyzing the distribution of join keys
print("Checking for skew in product_id:")
sales_df.groupBy("product_id").count().orderBy(col("count").desc()).show(5)


# 2. Handling skew with salting (adding randomness to distribute evenly)
def handle_skew_with_salting(skewed_df, skew_column, num_salts=5):
    """
    Handles skewed join keys by adding salting.
    Creates multiple copies of rows from the dimension table with different salt values.
    """
    # First identify the skewed keys
    skewed_keys = skewed_df.groupBy(skew_column) \
        .count() \
        .filter(col("count") > 100) \
        .select(skew_column) \
        .collect()

    skewed_keys_list = [row[0] for row in skewed_keys]

    if not skewed_keys_list:
        return skewed_df, product_df  # No skew detected

    # Add a salt column to the skewed fact table
    salted_fact = skewed_df.withColumn(
        "salt",
        F.when(
            col(skew_column).isin(skewed_keys_list),
            (F.rand() * num_salts).cast("int")
        ).otherwise(0)
    )

    # Replicate the dimension rows for skewed keys
    regular_dim = product_df.filter(~col("product_id").isin(skewed_keys_list))

    salted_dims = []
    for salt_value in range(num_salts):
        salted_dim = product_df.filter(col("product_id").isin(skewed_keys_list)) \
            .withColumn("salt", lit(salt_value))
        salted_dims.append(salted_dim)

    # Add salt=0 for regular keys
    regular_dim = regular_dim.withColumn("salt", lit(0))

    # Combine all the salted dimension pieces
    salted_dimension = regular_dim
    for dim in salted_dims:
        salted_dimension = salted_dimension.union(dim)

    return salted_fact, salted_dimension


# Example usage (not executed to keep the notebook focused)
# salted_sales, salted_products = handle_skew_with_salting(sales_df, "product_id")
# skew_handled_join = salted_sales.join(
#     salted_products,
#     (salted_sales.product_id == salted_products.product_id) &
#     (salted_sales.salt == salted_products.salt)
# )

# 3. Use Spark's built-in skew handling (AQE)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")

# With these settings, Spark will automatically split and handle skewed partitions

#############################################################
# SECTION 10: MONITORING AND DEBUGGING JOINS
#############################################################

"""
Monitoring and debugging join operations is essential for optimization.
Here are techniques to understand what's happening:
"""

# 1. Explain plans
print("Detailed explain plan:")
broadcast_join.explain(extended=True)

# 2. Use the Spark UI
"""
The Spark UI provides valuable insights for debugging join issues:
- Look at the DAG visualization to understand the execution flow
- Check the 'Stages' tab to identify bottlenecks in join operations
- Examine shuffle read/write metrics to detect excessive data movement
- Look for 'data skew' by comparing task durations in the same stage
- Check memory usage to identify potential OOM errors due to large shuffles

Access the UI at http://localhost:4040 when running locally.
"""

# 3. Event logging and history server
print("Enable event logging for persistent history:")


# spark.conf.set("spark.eventLog.enabled", "true")
# spark.conf.set("spark.eventLog.dir", "/path/to/event/logs")

# 4. Collect execution metrics
def analyze_join_performance(df):
    # Force execution and measure time
    import time
    start = time.time()
    count = df.count()  # Force execution
    duration = time.time() - start

    # Get the execution plan
    plan = df._jdf.queryExecution().executedPlan().toString()

    print(f"Row count: {count}")
    print(f"Execution time: {duration:.2f} seconds")
    print("Join strategy used:")

    # Look for specific join strategies in the plan
    if "BroadcastHashJoin" in plan:
        print("Broadcast Hash Join was used")
    elif "SortMergeJoin" in plan:
        print("Sort Merge Join was used")
    elif "ShuffledHashJoin" in plan:
        print("Shuffled Hash Join was used")

    return duration


# Example usage
print("Analyzing broadcast join performance:")
broadcast_time = analyze_join_performance(broadcast_join)

# Reset broadcast threshold to force sort-merge join
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
sort_merge_join = sales_df.join(product_df, "product_id")

print("Analyzing sort-merge join performance:")
sort_merge_time = analyze_join_performance(sort_merge_join)

# Reset to default
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")

# 5. Debugging common join issues
"""
Common join issues and solutions:

1. OutOfMemoryError during join:
   - Reduce broadcast threshold
   - Switch to sort-merge join
   - Increase executor memory
   - Filter data before join

2. Slow performance due to skew:
   - Enable AQE skew join handling
   - Implement salting technique
   - Filter data before join
   - Pre-aggregate on join key before joining

3. Cartesian join warning:
   - Check join condition - did you forget a join condition?
   - Verify data quality - are join keys present in both tables?
   - Consider if you actually need a cross join

4. Different result counts than expected:
   - Check for null values in join columns
   - Verify if you need inner vs outer join
   - Look for duplicate keys in tables


SUMMARY OF BEST PRACTICES FOR FACT-DIMENSION JOINS:

1. BROADCASTING:
   - Always broadcast dimension tables (explicitly with broadcast() or via hints)
   - Increase spark.sql.autoBroadcastJoinThreshold to accommodate larger dimensions
   - Cache dimension tables in memory

2. DATA PREPARATION:
   - Apply column pruning before joins (select only needed columns)
   - Apply filters before joins to reduce data size
   - Consider partitioning the fact table by common join/filter keys
   - For batch processing, consider bucketing tables by join keys

3. JOIN STRATEGY:
   - Use left joins to keep all fact records (unless inner is required)
   - Choose a good join order (most selective first)
   - Consider denormalizing very small dimensions into the fact table

4. HANDLING SKEW:
   - Enable AQE and skew join handling
   - For extreme skew, implement salting techniques
   - Consider pre-aggregation before joining for analytics queries

5. PERFORMANCE TUNING:
   - Monitor the Spark UI to identify bottlenecks
   - Tune executor memory and cores based on dimension table sizes
   - Adjust shuffle partitions for optimal parallelism
   - Use predicate pushdown when applicable

6. STORAGE OPTIMIZATION:
   - Use appropriate file formats (Parquet, ORC)
   - Use appropriate compression (Snappy, Zstd)
   - Consider columnar formats for dimension tables

7. ADVANCED TECHNIQUES FOR EXTREME SCALE:
   - Consider bloom filter joins
   - Implement bucketing on join keys
   - Use Z-ordering for multi-dimensional data clustering
   - For very large dimensions, consider broadcast-less techniques like Spark HyperLogLog

SPECIFIC RECOMMENDATIONS FOR 500GB FACT TABLE WITH SMALL DIMENSIONS:
1. Broadcast all dimension tables explicitly
2. Cache all dimension tables in memory
3. Enable adaptive query execution
4. Use column pruning aggressively
5. Pre-filter data when possible
6. For regular queries, create optimized views
7. Monitor and tune based on actual execution patterns
"""

# Clean up
spark.catalog.clearCache()
print("PySpark Joins and Optimization Guide completed.")

# Don't forget to stop the Spark session when completely done
# spark.stop()
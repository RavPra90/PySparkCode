from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Create Spark session
spark = SparkSession.builder.appName("WindowFunctionsDemo").getOrCreate()

# Sample data
data = [
    (101, "Alice Johnson", "Engineering", 95, 5, 120000),
    (102, "Bob Smith", "Engineering", 87, 3, 95000),
    (103, "Carol Davis", "Engineering", 92, 4, 110000),
    (104, "Eva Brown", "Engineering", 95, 6, 125000),

    (201, "Frank Miller", "Sales", 88, 4, 75000),
    (202, "Grace Lee", "Sales", 94, 5, 85000),
    (203, "Henry Taylor", "Sales", 99, 3, 90000),
    (204, "Iris Chen", "Sales", 91, 6, 90000),

    (301, "Kate Wilson", "Marketing", 89, 3, 65000),
    (302, "Liam Garcia", "Marketing", 83, 2, 55000),
    (303, "Mia Rodriguez", "Marketing", 89, 4, 68000),
    (304, "Noah Martinez", "Marketing", 85, 3, 60000),
]

columns = ["employee_id", "employee_name", "department", "performance_score", "years_experience", "salary"]

df = spark.createDataFrame(data, columns)
df.show()

# Define window specification - partition by department, order by performance_score descending
window_spec = Window.partitionBy("department").orderBy(col("performance_score").desc())

# Apply all three window functions
result_df = df.select(
    "*",
    # ROW_NUMBER: Sequential numbering (1, 2, 3, 4, 5...)
    row_number().over(window_spec).alias("row_num"),

    # RANK: Traditional ranking with gaps (1, 2, 2, 4, 5...)
    rank().over(window_spec).alias("rank"),

    # DENSE_RANK: Dense ranking without gaps (1, 2, 2, 3, 4...)
    dense_rank().over(window_spec).alias("dense_rank")
).orderBy("department", "row_num")

print("=== COMPLETE RANKING ANALYSIS ===")
result_df.show(50, truncate=False)

# Analysis 1: Top 2 performers in each department (using row_number)
print("=== TOP 3 PERFORMERS PER DEPARTMENT ===")
top_performers = result_df.filter(col("row_num") <= 2).select(
    "department", "employee_name", "performance_score", "row_num"
).orderBy("department", "row_num")

top_performers.show(truncate=False)

# Analysis 2: Count employees by rank in each department
print("\n=== RANK DISTRIBUTION BY DEPARTMENT ===")
rank_distribution = result_df.groupBy("department", "rank").agg(
    count("*").alias("employees_count"),
    collect_list("employee_name").alias("employees")
).orderBy("department", "rank")

rank_distribution.show(truncate=False)

# Analysis 3: Identify tied scores and ranking differences
print("\n=== TIED SCORES ANALYSIS ===")
tied_scores = result_df.filter(col("rank") != col("row_num")).select(
    "department", "employee_name", "performance_score",
    "row_num", "rank", "dense_rank",
    (col("rank") - col("dense_rank")).alias("rank_gap")
).orderBy("department", "performance_score")

tied_scores.show(truncate=False)

# Analysis 4: Departments with most ties
print("\n=== DEPARTMENTS WITH PERFORMANCE TIES ===")
ties_by_dept = result_df.groupBy("department").agg(
    count(when(col("rank") != col("row_num"), 1)).alias("employees_with_ties"),
    countDistinct("performance_score").alias("unique_scores"),
    count("*").alias("total_employees")
).withColumn(
    "tie_percentage",
    round((col("employees_with_ties") / col("total_employees")) * 100, 2)
).orderBy(col("tie_percentage").desc())

ties_by_dept.show(truncate=False)

# Analysis 5: Advanced - Show ranking comparison side by side
print("\n=== RANKING METHODS COMPARISON ===")
comparison = result_df.select(
    "department", "employee_name", "performance_score",
    "row_num", "rank", "dense_rank"
).orderBy("department", col("performance_score").desc())

comparison.show(50, truncate=False)
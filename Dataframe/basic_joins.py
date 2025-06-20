from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col

# Initialize Spark Session with optimized configuration
spark = SparkSession.builder \
    .appName("Comprehensive Join Operations") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Define schema for employees table
employees_schema = StructType([
    StructField("emp_id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("dept_id", IntegerType(), True),
    StructField("salary", IntegerType(), True),
    StructField("manager_id", IntegerType(), True)
])

# Define schema for departments table
departments_schema = StructType([
    StructField("dept_id", IntegerType(), True),
    StructField("dept_name", StringType(), True),
    StructField("location", StringType(), True),
    StructField("budget", IntegerType(), True)
])

# Create employees dataset with strategic edge cases
employees_data = [
    (1, "Alice", 10, 75000, 3),        # Valid employee with manager
    (2, "Bob", 20, 65000, None),       # Employee with null manager
    (3, "Carol", None, 85000, 1),      # Employee with null department
    (4, "David", 10, 70000, 1),        # Duplicate dept_id scenario
    (5, "Eva", 30, 60000, 2),          # Employee in non-existing dept
    (None, "Frank", 10, 55000, 3)      # Employee with null emp_id
]

# Create departments dataset with strategic edge cases
departments_data = [
    (10, "Engineering", "New York", 500000),     # Has employees
    (20, "Marketing", "California", 300000),     # Has one employee
    (None, "Unknown Dept", "Remote", 100000),    # Department with null ID
    (40, "Research", "Boston", 250000),          # No employees assigned
    (50, "Operations", "Chicago", 180000),       # No employees assigned
    (10, "Software Dev", "Austin", 450000)       # Duplicate dept_id
]

# Create DataFrames from the datasets
employees_df = spark.createDataFrame(employees_data, employees_schema)
departments_df = spark.createDataFrame(departments_data, departments_schema)

print("EMPLOYEES DATASET:")
employees_df.show()

print("DEPARTMENTS DATASET:")
departments_df.show()

# Problem: Show employees assigned to existing departments with dept_name
# Solution: Inner Join - Show only records with matching keys in both tables
print("1. Show employees assigned to existing departments with dept_name - INNER JOIN")
inner_join = employees_df.join(departments_df, "dept_id", "inner")
inner_join.select("name", "dept_name", "location", "salary").show()

# Problem: Show all employees along with their department info if available
# Solution: Left Join - Show all left table records with right matches where available
print("2. Show all employees along with their department info if available - LEFT JOIN ")
left_join = employees_df.join(departments_df, "dept_id", "left")
left_join.select("name", "dept_name", "location", "salary").show()

# Problem: Show all departments along with their employees if any
# Solution: Right Join - Show all right table records with left matches where available
print("3. Show all departments along with their employees if any - RIGHT JOIN")
right_join = employees_df.join(departments_df, "dept_id", "right")
right_join.select("dept_name", "name", "location", "salary").show()

# Problem: Show every employee and every department including those without a match
# Solution: Full Outer Join - Show union of all records from both tables with nulls for non-matches
print("4. Show every employee and every department including those without a match - FULL OUTER JOIN")
full_join = employees_df.join(departments_df, "dept_id", "full_outer")
full_join.select("name", "dept_name", "location", "salary").show()

# Problem: Show every possible pairing of employees and departments
# Solution: Cross Join - Show cartesian product combining every row from both tables
print("5. Show every possible pairing of employees and departments- CROSS JOIN")
# Use clean data and limit to prevent memory issues
emp_clean = employees_df.filter(col("emp_id").isNotNull()).limit(3)
dept_clean = departments_df.filter(col("dept_id").isNotNull()).limit(2)
cross_join = emp_clean.crossJoin(dept_clean)
cross_join.select("name", "dept_name", "location").show()

# Problem: Show employees who belong to at least one department
# Solution: Left Semi Join - Show left table records that have matches in right table
print("6. Show employees who belong to at least one department - LEFT SEMI JOIN")
semi_join = employees_df.join(departments_df, "dept_id", "left_semi")
semi_join.select("name", "dept_id", "salary").show()

# Problem: Show employees who aren't assigned to any department
# Solution: Left Anti Join - Show left table records that have no matches in right table
print("7. Show employees who aren't assigned to any department - LEFT ANTI JOIN ")
anti_join = employees_df.join(departments_df, "dept_id", "left_anti")
anti_join.select("name", "dept_id", "salary").show()

# Problem: Show each employee alongside their manager from the same table
# Solution: Self Join - Join table with itself using different aliases for hierarchical relationships
print("8. Show each employee alongside their manager from the same table- SELF JOIN ")
# Create manager reference by aliasing the same employees table
managers_df = employees_df.select(
    col("emp_id").alias("mgr_id"),
    col("name").alias("manager_name"),
    col("dept_id").alias("mgr_dept_id")
)
# Join employees with their managers using manager_id
self_join = employees_df.join(
    managers_df,
    employees_df.manager_id == managers_df.mgr_id,
    "left"
)
self_join.select("name", "manager_name", "salary", "dept_id").show()

# Problem: Show employees and departments matched automatically on common columns
# Solution: Natural Join - Automatic join on all columns with identical names
print("9. Show employees and departments matched automatically on common columns - NATURAL JOIN ")
natural_join = employees_df.join(departments_df, "dept_id")
natural_join.select("name", "dept_name", "location", "salary").show()

# Problem: Show employees and departments where dept_id values are equal
# Solution: Equi Join - Explicit equality condition with full control over join logic
print("10. Show employees and departments where dept_id values are equal- EQUI JOIN ")
equi_join = employees_df.alias("emp").join(
    departments_df.alias("dept"),
    col("emp.dept_id") == col("dept.dept_id"),
    "inner"
)
equi_join.select("emp.name", "dept.dept_name", "dept.location", "emp.salary").show()

# Stop Spark session to free resources
spark.stop()
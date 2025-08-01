from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, asc, row_number, rank, dense_rank, lit
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("RankingFunctionsDemo").getOrCreate()

# Create diverse dataset with ties and edge cases for comprehensive testing
data = [
    ("Alice Johnson", "Engineering", 95, 120000),  # Top performer, top salary
    ("Bob Smith", "Engineering", 88, 110000),  # Good performer, good salary
    ("Charlie Brown", "Engineering", 88, 110000),  # SAME as Bob - performance AND salary tie
    ("Diana Prince", "Engineering", 92, 105000),  # TOP performer but LOWER salary than others

    ("Eva Martinez", "Sales", 92, 130000),  # Top performer, TOP salary
    ("Frank Wilson", "Sales", 78, 95000),  # Low performer, low salary
    ("Grace Lee", "Sales", 92, 95000),  # TOP performance but LOWER salary than Eva
    ("Henry Davis", "Sales", 75, 85000),  # Lower performer, lower salary

    ("Ivy Chen", "Marketing", 89, 108000),  # Top performer, top salary
    ("Jack Turner", "Marketing", 89, 95000),  # TOP performance but LOWER salary than Ivy
    ("Kelly Moore", "Marketing", 82, 110000),  # Lower performance but HIGH salary
    ("Liam Taylor", "Marketing", 82, 90000)  # SAME performance as Kelly, lower salary
]

columns = ["employee_name", "department", "performance_score", "salary"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

print("=== ORIGINAL DATASET ===")
df.show()

# Define window specifications for different ranking scenarios

# Window partitioned by department, ordered by performance score ONLY (to show true ties)
window_perf_dept = Window.partitionBy("department").orderBy(desc("performance_score"))

# Window partitioned by department, ordered by salary ONLY (to show true ties)
window_salary_dept = Window.partitionBy("department").orderBy(desc("salary"))

# Window for company-wide performance ranking (ONLY by performance to show true tiers)
window_perf_company = Window.partitionBy(lit("1")).orderBy(desc("performance_score"))

print("\n=== RANKING COMPARISON SIDE BY SIDE ===")
# Show all ranking functions side by side for performance within department
comparison_df = df.select(
    "employee_name", "department", "performance_score", "salary",
    # ROW_NUMBER: Always assigns unique sequential numbers (1,2,3,4...)
    row_number().over(window_perf_dept).alias("row_num_perf"),
    # RANK: Assigns same rank to ties, skips next rank (1,1,3,4...)
    rank().over(window_perf_dept).alias("rank_perf"),
    # DENSE_RANK: Assigns same rank to ties, no gaps (1,1,2,3...)
    dense_rank().over(window_perf_dept).alias("dense_rank_perf")
)
comparison_df.orderBy("department", desc("performance_score")).show()

print("=== Q1: TOP 2 PERFORMERS IN EACH DEPARTMENT (ROW_NUMBER) ===")
# ROW_NUMBER ensures exactly 2 employees per department, even with ties
q1_result = df.select(
    "employee_name", "department", "performance_score", "salary",
    row_number().over(window_perf_dept).alias("performance_rank")
).filter(col("performance_rank") <= 2)  # Filter top 2 by row number

q1_result.orderBy("department", "performance_rank").show()

print("=== Q2: EMPLOYEES WITH TOP 3 SALARY TIES IN EACH DEPARTMENT (RANK) ===")
# RANK handles ties by giving same rank, may return more than 3 employees
q2_result = df.select(
    "employee_name", "department", "performance_score", "salary",
    rank().over(window_salary_dept).alias("salary_rank")
).filter(col("salary_rank") <= 3)  # Include all employees tied for top 3

q2_result.orderBy("department", "salary_rank").show()

print("=== Q3: DISTINCT PERFORMANCE TIERS COMPANY-WIDE (DENSE_RANK) ===")
# DENSE_RANK shows continuous performance tiers without gaps
q3_result = df.select(
    "employee_name", "department", "performance_score",
    dense_rank().over(window_perf_company).alias("performance_tier")
).orderBy("performance_tier")

q3_result.show()

# Count unique tiers
tier_count = q3_result.select("performance_tier").distinct().count()
print(f"Total unique performance tiers: {tier_count}")

print("=== Q4: HIGH DEPT PERFORMANCE TIER BUT LOW DEPT SALARY TIER ===")
# Find undervalued high performers: top performance but lower salary within department
q4_prep = df.select(
    "employee_name", "department", "performance_score", "salary",
    dense_rank().over(window_perf_dept).alias("dept_perf_tier"),
    dense_rank().over(window_salary_dept).alias("dept_salary_tier")
)

# Top performance tier (1) but not in top salary tier (>1) within same department
q4_result = q4_prep.filter(
    (col("dept_perf_tier") == 1) &  # Highest performance tier in dept
    (col("dept_salary_tier") > 1)  # Not highest salary tier in dept
)

print("Undervalued High Performers (Top performance tier but lower salary tier):")
q4_result.orderBy("department", "dept_perf_tier").show()

print("=== Q5: TOP DEPT SALARY TIER BUT LOW DEPT PERFORMANCE TIER ===")
# Find overpaid underperformers: top salary but lower performance within department
q5_result = q4_prep.filter(
    (col("dept_salary_tier") == 1) &  # Highest salary tier in dept
    (col("dept_perf_tier") > 1)  # Not highest performance tier in dept
)

print("Overpaid Underperformers (Top salary tier but lower performance tier):")
q5_result.orderBy("department", "dept_salary_tier").show()

# Stop Spark session
spark.stop()

"""
KEY STEPS IN THE CODING APPROACH:
1. Create dataset with REAL ties in both performance and salary within departments
2. Define Window specifications WITHOUT secondary sort to show true ranking behavior
3. Demonstrate how ROW_NUMBER, RANK, and DENSE_RANK handle ties differently
4. Use appropriate ranking functions for each business scenario
5. Focus on department-level vs company-level comparisons

EDGE CASES COVERED:
- Perfect ties: Bob & Charlie (88 performance, 110K salary)
- Triple ties: Eva, Frank, Grace (all 95K salary in Sales)
- Performance ties with same salary: Ivy & Jack (89 performance, 108K salary)
- Performance ties with different salary: Kelly & Liam (82 performance, different salaries)

DEVELOPER TIPS:
- Remove secondary sort columns to see true tie behavior
- ROW_NUMBER assigns arbitrary unique numbers to ties (1,2,3,4)
- RANK gives same rank to ties, skips next numbers (1,1,3,4)  
- DENSE_RANK gives same rank to ties, no gaps (1,1,2,3)
- Use DENSE_RANK for tier analysis to avoid gaps
- Test scenarios with actual ties in your dataset
- Verify your ranking logic matches business requirements
"""
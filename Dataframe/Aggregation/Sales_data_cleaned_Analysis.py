# 🚀 Real-World Sales Data Cleaning & Performance Analysis with PySpark

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re

# Initialize Spark Session - Entry point for all Spark operations
spark = SparkSession.builder \
    .appName("SalesDataCleaningAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

print("🔧 STEP 1: CREATING DIRTY DATASET")
print("=" * 50)

# Create intentionally messy/dirty sales dataset - Real-world scenarios
dirty_sales_data = [
    ("S001", "  John Doe  ", "electronics", "Laptop Pro", "1200.50", "2024-01-15", "NORTH", "john@email.com"),
    ("S002", "", "Clothing", "winter jacket", "80", "2024/01/16", "south", "sarah@email.com"),
    ("", "Mike Johnson", "ELECTRONICS", "iPhone 15", "800.00", "15-01-2024", "East", "mike@company.com"),
    ("S004", "John Doe", "home & garden", "Office Chair", "150.99", "2024-01-18", "North", "john@email.com"),
    ("S005", "Sarah Wilson", "Electronics", "iPad", "300", "2024-01-32", "South", ""),  # Invalid date
    ("S006", "mike johnson", "clothing", "Running Shoes", "abc", "2024-01-20", "EAST", "mike@company.com"),  # Invalid amount
    ("S007", None, "Electronics", "Smart Watch", "250.00", "", "north", "test@test.com"),  # Empty date
    ("S008", "Lisa Brown", "Home", "Dining Table", "-400.00", "2024-01-22", "West", "lisa@email.com"),  # Negative amount
    ("S009", "SARAH WILSON", "electronics", "Headphones", "100.50", "2024-01-23", "", "sarah@email.com"),  # Empty region
    ("S010", "Mike Johnson", "Home", "Table Lamp", "75.00", "2024-01-24", "East", "mike@company.com"),
    ("S011", "john doe", "Electronics", "Tablet", "299.99", "2024-01-25", "North", "john@email.com"),
    (None, "Emma Davis", "Beauty", "Skincare Set", "120.00", "2024-01-26", "West", "emma@email.com"),  # Null sale_id
    ("S013", "Robert Smith", "electronics", "Gaming Mouse", "45.99", "2024-01-27", "North", "robert@email.com"),
    ("S014", "  ", "Clothing", "Jeans", "60.00", "2024-01-28", "South", "anonymous@email.com"),  # Empty salesperson
    ("S015", "Lisa Brown", "HOME", "Bookshelf", "180.00", "2024-01-29", "west", "lisa@email.com")
]

# Create DataFrame without schema to see raw messy data
raw_df = spark.createDataFrame(dirty_sales_data,
    ["sale_id", "salesperson", "category", "product", "amount", "sale_date", "region", "email"])

print("📊 Original Dirty Dataset:")
raw_df.show(15, truncate=False)
print(f"Total Records: {raw_df.count()}")

print("\n🔍 STEP 2: DATA QUALITY ASSESSMENT")
print("=" * 50)

# Check data quality issues
print("Data Quality Issues Found:")
print(f"• Null sale_id: {raw_df.filter(col('sale_id').isNull()).count()}")
print(f"• Empty sale_id: {raw_df.filter(col('sale_id') == '').count()}")
print(f"• Null/Empty salesperson: {raw_df.filter(col('salesperson').isNull() | (trim(col('salesperson')) == '')).count()}")
print(f"• Empty region: {raw_df.filter(col('region') == '').count()}")
print(f"• Empty email: {raw_df.filter(col('email') == '').count()}")

# Check for inconsistent data formats
print(f"• Inconsistent categories: {raw_df.select('category').distinct().count()} unique values")
raw_df.select("category").distinct().show()

print("\n🧹 STEP 3: DATA CLEANING OPERATIONS")
print("=" * 50)

# Step 3.1: Handle missing sale_ids by generating new ones
print("3.1 Generating missing sale_ids...")
from pyspark.sql.window import Window

# Create window for row numbering
window_spec = Window.orderBy("salesperson", "sale_date")

cleaned_df = raw_df.withColumn("row_num", row_number().over(window_spec)) \
    .withColumn("sale_id",
        when(col("sale_id").isNull() | (col("sale_id") == ""),
             concat(lit("S"), lpad(col("row_num") + 100, 3, "0")))
        .otherwise(col("sale_id"))
    ).drop("row_num")

# Step 3.2: Clean salesperson names
print("3.2 Cleaning salesperson names...")
cleaned_df = cleaned_df.withColumn("salesperson",
    when(col("salesperson").isNull() | (trim(col("salesperson")) == ""), "Unknown")
    .otherwise(
        # Convert to proper case and trim whitespace
        initcap(trim(regexp_replace(col("salesperson"), "\\s+", " ")))
    )
)

# Step 3.3: Standardize categories
print("3.3 Standardizing categories...")
cleaned_df = cleaned_df.withColumn("category",
    when(upper(col("category")).contains("ELECTRONIC"), "Electronics")
    .when(upper(col("category")).contains("CLOTH"), "Clothing")
    .when(upper(col("category")).contains("HOME"), "Home")
    .when(upper(col("category")).contains("BEAUTY"), "Beauty")
    .otherwise(initcap(col("category")))
)

# Step 3.4: Clean and validate amounts
print("3.4 Cleaning and validating amounts...")
cleaned_df = cleaned_df.withColumn("amount_clean",
    when(col("amount").rlike("^-?\\d+(\\.\\d+)?$"), col("amount").cast("double"))
    .otherwise(None)
) \
.withColumn("amount",
    when((col("amount_clean").isNull()) | (col("amount_clean") <= 0), None)
    .otherwise(col("amount_clean"))
) \
.drop("amount_clean")

# Step 3.5: Clean and validate dates
print("3.5 Cleaning and validating dates...")
cleaned_df = cleaned_df.withColumn("sale_date_clean",
    # Try multiple date formats
    coalesce(
        to_date(col("sale_date"), "yyyy-MM-dd"),
        to_date(col("sale_date"), "yyyy/MM/dd"),
        to_date(col("sale_date"), "dd-MM-yyyy")
    )
) \
.withColumn("sale_date",
    when(col("sale_date_clean").isNull(), None)
    .otherwise(col("sale_date_clean"))
) \
.drop("sale_date_clean")

# Step 3.6: Standardize regions
print("3.6 Standardizing regions...")
cleaned_df = cleaned_df.withColumn("region",
    when(col("region") == "", "Unknown")
    .otherwise(initcap(trim(col("region"))))
)

# Step 3.7: Validate email formats
print("3.7 Validating email formats...")
email_pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
cleaned_df = cleaned_df.withColumn("email",
    when((col("email") == "") | (~col("email").rlike(email_pattern)), None)
    .otherwise(lower(trim(col("email"))))
)

print("\n✅ STEP 4: CLEANED DATASET RESULTS")
print("=" * 50)

# Filter out records with critical missing data
final_clean_df = cleaned_df.filter(
    col("amount").isNotNull() &
    col("sale_date").isNotNull() &
    (col("salesperson") != "Unknown")
)

print("📊 Cleaned Dataset:")
final_clean_df.show(truncate=False)
print(f"Records after cleaning: {final_clean_df.count()}")
print(f"Records removed: {raw_df.count() - final_clean_df.count()}")

# Cache the cleaned dataset for multiple operations
final_clean_df.cache()

print("\n📈 STEP 5: DATA QUALITY VALIDATION")
print("=" * 50)

print("Data Quality After Cleaning:")
print(f"• Null amounts: {final_clean_df.filter(col('amount').isNull()).count()}")
print(f"• Null dates: {final_clean_df.filter(col('sale_date').isNull()).count()}")
print(f"• Unknown salesperson: {final_clean_df.filter(col('salesperson') == 'Unknown').count()}")
print(f"• Invalid emails: {final_clean_df.filter(col('email').isNull()).count()}")

# Show data distribution
print("\nCleaned Categories Distribution:")
final_clean_df.groupBy("category").count().orderBy(desc("count")).show()

print("\n📊 STEP 6: SALES PERFORMANCE ANALYSIS")
print("=" * 50)

# 6.1: SALESPERSON PERFORMANCE ANALYSIS
print("6.1 Salesperson Performance Analysis:")
salesperson_performance = final_clean_df.groupBy("salesperson") \
    .agg(
        sum("amount").alias("total_sales"),
        count("sale_id").alias("total_transactions"),
        avg("amount").alias("avg_sale_value"),
        max("amount").alias("highest_sale"),
        min("amount").alias("lowest_sale"),
        countDistinct("category").alias("categories_sold")
    ) \
    .withColumn("avg_sale_value", round(col("avg_sale_value"), 2)) \
    .orderBy(desc("total_sales"))

salesperson_performance.show()

# 6.2: CATEGORY PERFORMANCE ANALYSIS
print("\n6.2 Category Performance Analysis:")
total_revenue = final_clean_df.agg(sum("amount")).collect()[0][0]

category_analysis = final_clean_df.groupBy("category") \
    .agg(
        sum("amount").alias("category_revenue"),
        count("*").alias("units_sold"),
        avg("amount").alias("avg_price_point"),
        countDistinct("salesperson").alias("salespeople_count")
    ) \
    .withColumn("revenue_share_pct",
                round((col("category_revenue") / total_revenue) * 100, 2)) \
    .withColumn("avg_price_point", round(col("avg_price_point"), 2)) \
    .orderBy(desc("category_revenue"))

category_analysis.show()

# 6.3: REGIONAL PERFORMANCE ANALYSIS
print("\n6.3 Regional Performance Analysis:")
regional_performance = final_clean_df.groupBy("region") \
    .agg(
        sum("amount").alias("regional_sales"),
        count("*").alias("total_sales_count"),
        countDistinct("salesperson").alias("active_salespeople"),
        avg("amount").alias("avg_regional_sale"),
        countDistinct("category").alias("categories_available")
    ) \
    .withColumn("avg_regional_sale", round(col("avg_regional_sale"), 2)) \
    .orderBy(desc("regional_sales"))

regional_performance.show()

# 6.4: TIME-BASED ANALYSIS
print("\n6.4 Daily Sales Trend Analysis:")
daily_sales = final_clean_df.groupBy("sale_date") \
    .agg(
        sum("amount").alias("daily_revenue"),
        count("*").alias("daily_transactions"),
        countDistinct("salesperson").alias("active_salespeople")
    ) \
    .withColumn("avg_transaction_value",
                round(col("daily_revenue") / col("daily_transactions"), 2)) \
    .orderBy("sale_date")

daily_sales.show()

# 6.5: ADVANCED ANALYTICS - TOP PERFORMERS BY CATEGORY
print("\n6.5 Advanced Analytics - Top Performers by Category:")
window_cat = Window.partitionBy("category").orderBy(desc("amount"))

top_sales_by_category = final_clean_df.withColumn("rank_in_category",
                                                 row_number().over(window_cat)) \
    .filter(col("rank_in_category") <= 2) \
    .select("category", "salesperson", "product", "amount", "rank_in_category") \
    .orderBy("category", "rank_in_category")

top_sales_by_category.show()

# 6.6: PERFORMANCE TIER CLASSIFICATION
print("\n6.6 Sales Performance Tier Classification:")
performance_tiers = salesperson_performance \
    .withColumn("performance_tier",
        when(col("total_sales") >= 800, "🏆 Top Performer")
        .when(col("total_sales") >= 400, "🥈 High Performer")
        .when(col("total_sales") >= 200, "🥉 Medium Performer")
        .otherwise("📈 Developing")
    ) \
    .withColumn("efficiency_score",
        round(col("total_sales") / col("total_transactions"), 2)
    )

performance_tiers.select("salesperson", "total_sales", "total_transactions",
                        "efficiency_score", "performance_tier").show()

# 6.7: COMPREHENSIVE BUSINESS INSIGHTS
print("\n6.7 Key Business Insights Summary:")
print("=" * 40)

# Calculate key metrics
total_sales = final_clean_df.agg(sum("amount")).collect()[0][0]
total_transactions = final_clean_df.count()
unique_salespeople = final_clean_df.select("salesperson").distinct().count()
top_category = category_analysis.select("category").first()[0]
top_performer = salesperson_performance.select("salesperson").first()[0]

print(f"💰 Total Revenue: ${total_sales:,.2f}")
print(f"📊 Total Transactions: {total_transactions}")
print(f"👥 Active Salespeople: {unique_salespeople}")
print(f"🏆 Top Category: {top_category}")
print(f"🥇 Top Performer: {top_performer}")
print(f"📈 Average Transaction: ${total_sales/total_transactions:.2f}")

# Cleanup
final_clean_df.unpersist()
spark.stop()

"""
🔍 STEP-BY-STEP ANALYSIS PERFORMED:

📋 DATA CLEANING PIPELINE:
1. **Missing Value Imputation**: Generated sale_ids, handled null salesperson names
2. **Data Standardization**: Normalized categories, regions, and name formats
3. **Data Type Conversion**: Converted amounts to numeric, standardized date formats
4. **Data Validation**: Validated email formats, removed negative amounts
5. **Quality Filtering**: Removed records with critical missing information

🎯 AGGREGATION OPERATIONS:
1. **Salesperson Analysis**: Revenue, transaction count, average sale value, performance metrics
2. **Category Analysis**: Revenue distribution, market share, price point analysis
3. **Regional Analysis**: Geographic performance, resource allocation insights
4. **Temporal Analysis**: Daily trends, transaction patterns
5. **Advanced Analytics**: Window functions for ranking, performance classification

⚠️ REAL-WORLD EDGE CASES HANDLED:
• Mixed date formats (yyyy-MM-dd, yyyy/MM/dd, dd-MM-yyyy)
• Inconsistent text casing and spacing
• Invalid numeric values (non-numeric strings, negative amounts)
• Missing critical identifiers (sale_id, salesperson)
• Inconsistent category naming conventions
• Invalid email formats
• Empty/null values across multiple columns

💡 PRODUCTION CONSIDERATIONS:
✅ **Data Quality Metrics**: Track cleaning success rates
✅ **Error Logging**: Log all data quality issues for monitoring  
✅ **Schema Evolution**: Handle new data formats gracefully
✅ **Performance Optimization**: Cache frequently used datasets
✅ **Data Lineage**: Track transformations for auditing
✅ **Validation Rules**: Implement business rule validations
"""

# LinkedIn Hashtags:
# #PySpark #DataCleaning #DataQuality #BigData #DataEngineering #ETL
# #SalesAnalytics #ApacheSpark #DataScience #BusinessIntelligence #Python
# #DataGovernance #DataTransformation #Analytics #RealWorldData
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, regexp_replace, when, split, size, coalesce, lit, length, isnan, isnull

# Start Spark session
spark = SparkSession.builder.appName("NullSafeTextCleaning").getOrCreate()

# Create realistic messy data with NULL values
messy_data = [
    ("001", "  john_DOE-smith  ", "JOHN.doe@Gmail.COM", "+1-(555)-123-4567"),
    ("002", "jane@MARTINEZ_rodriguez", "Jane_Smith@yahoo.com  ", "555.987.6543"),
    ("003", "Bob-O'Connor Wilson", "bob.oconnor@hotmail.com", "(555) 246-8135"),
    ("004", "María_González-Castro", "maria.gonzalez@company.co.uk", "+44-20-7946-0958"),
    ("005", None, None, None),  # Complete NULL record
    ("006", "Alice   Johnson@Brown", None, "1 234 567 8900"),  # Partial NULL
    ("007", "DAVID Thompson", "DAVID@test.COM", None),  # Partial NULL
    ("008", None, "sarah.williams@email.co", "555-123-4567"),  # Partial NULL
    ("009", "", "", ""),  # Empty strings
    ("010", "SingleName", "invalid-email", "123")  # Invalid formats
]

# Column names for our data
columns = ["id", "full_name", "email", "phone"]
df = spark.createDataFrame(messy_data, columns)

print("=== ORIGINAL DATA WITH NULLS ===")
df.show(truncate=False)

# STEP 1: Handle NULL values FIRST - this prevents errors in later steps
# coalesce() replaces null with empty string for safer string operations
# Null handling should be the first step in any data pipeline
df_null_safe = df.withColumn("full_name", coalesce(col("full_name"), lit(""))) \
                 .withColumn("email", coalesce(col("email"), lit(""))) \
                 .withColumn("phone", coalesce(col("phone"), lit("")))

# STEP 2: Clean the full_name column
# Remove extra spaces from beginning and end
df_step1 = df_null_safe.withColumn("full_name_trimmed", trim(col("full_name")))

# Replace special characters (_, @, -) with single space
df_step2 = df_step1.withColumn("full_name_no_special",
                               regexp_replace(col("full_name_trimmed"), "[_@-]", " "))

# Replace multiple spaces with single space
df_step3 = df_step2.withColumn("full_name_clean",
                               regexp_replace(col("full_name_no_special"), " +", " "))

# STEP 3: Split names properly
# Split the cleaned name by space to get name parts
df_step4 = df_step3.withColumn("name_parts", split(col("full_name_clean"), " "))

# Get first name - always the first part (if exists)
df_step5 = df_step4.withColumn("first_name",
                               when(length(trim(col("full_name_clean"))) > 0, col("name_parts")[0])
                               .otherwise(""))

# Get last name - smart logic based on number of parts
df_step6 = df_step5.withColumn("last_name",
                               when(length(trim(col("full_name_clean"))) == 0, "")  # Empty name
                               .when(size(col("name_parts")) == 1, "")  # Single name
                               .when(size(col("name_parts")) == 2, col("name_parts")[1])  # Two names
                               .otherwise(col("name_parts")[size(col("name_parts")) - 1]))  # Take last part

# STEP 4: Clean email addresses
df_step7 = df_step6.withColumn("email_clean",
                               lower(trim(col("email"))))

# STEP 5: Clean phone numbers
df_step8 = df_step7.withColumn("phone_clean",
                               regexp_replace(col("phone"), "[^0-9]", ""))

# STEP 6: VALIDATE RECORDS - Create validation rules
df_validated = df_step8.withColumn("record_status",
    # A record is VALID if:
    # 1. Has first_name (not empty)
    # 2. Has valid email format
    # 3. Has phone with at least 10 digits
    when(
        (length(col("first_name")) > 0) &  # Has first name
        (col("email_clean").rlike("^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$")) &  # Valid email format
        (length(col("phone_clean")) >= 10)  # Valid phone length
    ,"VALID")
    .otherwise("INVALID")
)

# STEP 7: Create final clean dataset
df_final = df_validated.select(
    "id",
    "first_name",
    "last_name",
    "email_clean",
    "phone_clean",
    "record_status"
)

print("\n=== FINAL CLEANED DATA WITH VALIDATION ===")
df_final.show(truncate=False)

# STEP 8: GENERATE VALIDATION COUNTS
print("\n=== VALIDATION SUMMARY ===")
validation_counts = df_final.groupBy("record_status").count().orderBy("record_status")
validation_counts.show()

# Get specific counts
total_records = df_final.count()
valid_records = df_final.filter(col("record_status") == "VALID").count()
invalid_records = df_final.filter(col("record_status") == "INVALID").count()

print(f" SUMMARY STATISTICS:")
print(f"Total Records: {total_records}")
print(f"Valid Records: {valid_records}")
print(f"Invalid Records: {invalid_records}")
print(f"Success Rate: {(valid_records/total_records)*100:.1f}%")


spark.stop()

# 🎯 KEY CONCEPTS EXPLAINED:

# 1. NULL SAFETY FIRST:
#    - Always handle NULLs before string operations
#    - coalesce(col, "") replaces NULL with empty string
#    - Prevents "NullPointerException" in production

# 2. VALIDATION LOGIC:
#    - Uses AND (&) operator for multiple conditions
#    - All conditions must be TRUE for "VALID" status
#    - Easy to modify validation rules as business needs change

# 3. REGEX FOR EMAIL:
#    - ^[a-z0-9._%+-]+  : starts with alphanumeric/special chars
#    - @[a-z0-9.-]+     : @ symbol followed by domain characters
#    - \\.[a-z]{2,}$    : ends with dot and 2+ letter extension

# 4. PHONE VALIDATION:
#    - Extracts only digits: [^0-9] removes non-numeric
#    - Checks length >= 10 for valid phone numbers
#    - Handles international and domestic formats

# 🛡️ PRODUCTION BENEFITS:
# - Data quality monitoring with clear metrics
# - Easy identification of problematic records
# - Audit trail for data cleaning decisions
# - Scalable validation framework

# 🚀 BUSINESS VALUE:
# - Improves downstream analytics accuracy
# - Reduces customer communication failures
# - Enables better data-driven decisions
# - Supports compliance and reporting requirements
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode_outer, expr, from_json, schema_of_json

# Initialize Spark Session
spark = SparkSession.builder.appName("Optimized_JSON_Flattening").getOrCreate()

# Sample nested JSON data
nested_json = """
{
  "event_id": "e123",
  "timestamp": "2025-05-19T10:30:00Z",
  "user": {
    "id": "u456",
    "name": "Jane Doe",
    "preferences": {
      "theme": "dark",
      "notifications": true
    }
  },
  "actions": [
    {
      "action_id": "a1",
      "type": "click",
      "elements": [
        {"element_id": "btn1", "name": "submit"},
        {"element_id": "btn2", "name": "cancel"}
      ]
    },
    {
      "action_id": "a2",
      "type": "view",
      "elements": [
        {"element_id": "img1", "name": "banner"}
      ]
    }
  ]
}
"""

# Create DataFrame from JSON
df = spark.read.json(spark.sparkContext.parallelize([nested_json]))

# Method 1: Using a single select statement with path expressions
# This flattens everything in one operation for simple cases
flattened_df = df.selectExpr(
    "event_id",
    "timestamp",
    "user.id as user_id",                               # Direct path expressions flatten nested structures
    "user.name as user_name",
    "user.preferences.theme as user_theme",
    "user.preferences.notifications as user_notifications"
)

# Method 2: For complex nested arrays with production-level controls
# This approach handles the parent-child relationships more explicitly
result_df = df.select(
    col("event_id"),
    col("timestamp"),
    col("user.id").alias("user_id"),
    col("user.name").alias("user_name"),
    col("user.preferences.theme").alias("user_theme"),
    col("user.preferences.notifications").alias("user_notifications"),
    col("actions")
).withColumn(
    "action", explode_outer(col("actions"))            # Use explode_outer to handle empty arrays
).select(
    col("event_id"),
    col("timestamp"),
    col("user_id"),
    col("user_name"),
    col("user_theme"),
    col("user_notifications"),
    col("action.action_id"),                           # Extract fields from exploded struct
    col("action.type").alias("action_type"),
    col("action.elements")
).withColumn(
    "element", explode_outer(col("elements"))          # Explode the nested array
).select(
    col("event_id"),
    col("timestamp"),
    col("user_id"),
    col("user_name"),
    col("user_theme"),
    col("user_notifications"),
    col("action_id"),
    col("action_type"),
    col("element.element_id"),
    col("element.name").alias("element_name")
)

result_df.show()


# Method 3: Using SQL expressions for dynamic handling
# Register as temp view to use SQL for complex transformations
df.createOrReplaceTempView("nested_data")

sql_flattened = spark.sql("""
    SELECT
        event_id,
        timestamp,
        user.id as user_id,
        user.name as user_name,
        user.preferences.theme as user_theme,
        user.preferences.notifications as user_notifications,
        act.action_id,
        act.type as action_type,
        elem.element_id,
        elem.name as element_name
    FROM nested_data
    LATERAL VIEW OUTER EXPLODE(actions) actions_table AS act      -- LATERAL VIEW is SQL's explode
    LATERAL VIEW OUTER EXPLODE(act.elements) elements_table AS elem
""")

# Display results
sql_flattened.show(truncate=False)



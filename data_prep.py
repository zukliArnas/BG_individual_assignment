from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, unix_timestamp, round, month, to_timestamp
)
from logger import get_logger


logger = get_logger("task.log")

def create_spark_session():
    return SparkSession.builder \
        .appName("Bike Rent Analysis") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()

def load_and_clean_data(spark, path):
    df = spark.read.option("header", True).csv(path)
    logger.info("Cleaning columns")

    total = df.count()
    unique = df.select("ride_id").distinct().count()

    df_clean = df \
        .withColumn("started_at", to_timestamp("started_at")) \
        .withColumn("ended_at", to_timestamp("ended_at")) \
        .withColumn("trip_duration_min", round((unix_timestamp("ended_at") - unix_timestamp("started_at")) / 60.0, 2)) \
        .withColumn("month", month("started_at")) \
        .filter(col("trip_duration_min").isNotNull()) \
        .filter(col("trip_duration_min") > 0) \
        .drop("start_station_id", "end_station_id", "ride_id")
    
    logger.info(f"ride_id unique: {total == unique}")
    if total == unique:
        df = df.drop("ride_id")

    logger.info(f"Rows before cleaning: {df.count()}")
    logger.info(f"Rows after cleaning: {df_clean.count()}")
    df_clean.show()

    return df_clean

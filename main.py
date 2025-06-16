import pandas as pd
from logger import get_logger
from data_prep import create_spark_session, load_and_clean_data
from utils_plots import *
from pyspark.sql.functions import avg, hour, col

logger = get_logger("task.log")
DIR = "/Users/arnaszuklija/Desktop/data_science_2/big_data/individual_assignment/bike_rent/*.csv"


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    df_clean = load_and_clean_data(spark, DIR)

    # Avg duration per user type
    df_clean.groupBy("member_casual").agg(avg("trip_duration_min")).show()

    # Monthly rides by user type
    df_clean.groupBy("month", "member_casual").count().orderBy("month").show(n=50)

    # Bike type usage
    df_clean.groupBy("rideable_type", "member_casual").count().show()

    # Peak usage hours
    df_clean.withColumn("hour", hour("started_at")).groupBy("hour").count().orderBy("hour").show(n=50)

    # Convert to pandas for plotting
    pandas_df = df_clean.select("trip_duration_min", "month", "member_casual").toPandas()
    df_clean.filter(col("start_lat").isNull() | col("end_lat").isNull()).count()

    # # Plot
    plot_monthly_trip_duration_histograms(pandas_df)
    plot_monthly_ride_counts(pandas_df)
    plot_ride_type_distribution(df_clean)
    generate_route_map_by_daytype(df_clean)
    get_avg_duration_by_bike_and_user(df_clean).show()
    
if __name__ == "__main__":
    main()

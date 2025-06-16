import matplotlib.pyplot as plt
import branca
import pandas as pd
import seaborn as sns
import folium
from folium import plugins
from pyspark.sql.functions import col, dayofweek, when, count
from logger import get_logger

logger = get_logger("task.log")

def plot_monthly_trip_duration_histograms(df):
    """
    Plots monthly histograms of trip duration counts, separated by user type.
    """
    # Optional: filter to focus on common range
    df = df[df["trip_duration_min"] < 300]

    # Create FacetGrid
    g = sns.FacetGrid(
        df,
        col="month",
        col_wrap=4,
        hue="member_casual",
        height=4,
        sharex=True,
        sharey=True
    )

    g.map(
        sns.histplot,
        "trip_duration_min",
        bins=40,
        stat="count",
        element="step",
        common_norm=False
    )

    g.set_axis_labels("Trip Duration (min)", "Ride Count")
    g.add_legend(title="User Type")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Monthly Trip Duration Histograms by User Type")
    plt.tight_layout()
    plt.show()

def plot_monthly_ride_counts(df):
    """
    Plots a grouped bar chart of total rides per month by user type.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Aggregate counts
    monthly_counts = df.groupby(["month", "member_casual"]).size().reset_index(name="count")

    # Sort month as integer (just in case)
    monthly_counts["month"] = monthly_counts["month"].astype(int)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=monthly_counts,
        x="month",
        y="count",
        hue="member_casual"
    )
    plt.title("Total Rides per Month by User Type")
    plt.xlabel("Month")
    plt.ylabel("Number of Rides")
    plt.legend(title="User Type")
    plt.tight_layout()
    plt.show()

def generate_route_map_by_daytype(df_routes, output_file="daytype_route_map.html", max_routes=10000):
    logger.info("generate_route_map_by_daytype() - Starting...")

    # Ensure timestamp is available
    df = df_routes.withColumn("day_of_week", dayofweek("started_at"))
    df = df.withColumn("day_type", when((col("day_of_week") == 1) | (col("day_of_week") == 7), "Weekend").otherwise("Weekday"))

    # Group & count
    df_grouped = df.groupBy(
        "start_lat", "start_lng", "end_lat", "end_lng", "member_casual", "day_type"
    ).agg(count("*").alias("route_count"))

    # Top 100 routes by count
    top_routes_df = df_grouped.orderBy(col("route_count").desc()).limit(100)
    top_routes_pd = top_routes_df.toPandas()

    sampled_df = df_grouped.sample(False, 0.005).toPandas()
    df_combined = pd.concat([sampled_df, top_routes_pd], ignore_index=True).drop_duplicates()

    df_combined = df_combined.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])

    # Create map
    m = folium.Map(location=[41.88, -87.63], zoom_start=12)

    # FeatureGroups for toggling
    fg_weekday_casual = folium.FeatureGroup(name="Weekday - Casual")
    fg_weekday_member = folium.FeatureGroup(name="Weekday - Member")
    fg_weekend_casual = folium.FeatureGroup(name="Weekend - Casual")
    fg_weekend_member = folium.FeatureGroup(name="Weekend - Member")
    fg_top_routes = folium.FeatureGroup(name="Top 100 Most Used Routes")

    for _, row in df_combined.iterrows():
        route = [(float(row["start_lat"]), float(row["start_lng"])), (float(row["end_lat"]), float(row["end_lng"]))]
        tooltip = f"{row['member_casual']} - {row['day_type']} ({row['route_count']})"
        is_top = (
            (top_routes_pd["start_lat"] == row["start_lat"]) &
            (top_routes_pd["start_lng"] == row["start_lng"]) &
            (top_routes_pd["end_lat"] == row["end_lat"]) &
            (top_routes_pd["end_lng"] == row["end_lng"])
        ).any()

        polyline = folium.PolyLine(
            locations=route,
            color="red" if is_top else (
                "blue" if row["day_type"] == "Weekday" and row["member_casual"] == "casual" else
                "green" if row["day_type"] == "Weekday" else
                "orange" if row["member_casual"] == "casual" else
                "purple"
            ),
            weight=3 if is_top else 2,
            opacity=0.6,
            tooltip=tooltip
        )

        if is_top:
            polyline.add_to(fg_top_routes)
        elif row["day_type"] == "Weekday" and row["member_casual"] == "casual":
            polyline.add_to(fg_weekday_casual)
        elif row["day_type"] == "Weekday" and row["member_casual"] == "member":
            polyline.add_to(fg_weekday_member)
        elif row["day_type"] == "Weekend" and row["member_casual"] == "casual":
            polyline.add_to(fg_weekend_casual)
        else:
            polyline.add_to(fg_weekend_member)

    # Add layers to map
    fg_weekday_casual.add_to(m)
    fg_weekday_member.add_to(m)
    fg_weekend_casual.add_to(m)
    fg_weekend_member.add_to(m)
    fg_top_routes.add_to(m)

    folium.LayerControl().add_to(m)

    # Optional: HTML legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 240px; height: 170px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding:10px;">
        <b>Legend</b><br>
        🔵 Weekday - Casual<br>
        🟢 Weekday - Member<br>
        🟠 Weekend - Casual<br>
        🟣 Weekend - Member<br>
        🔴 Top 100 Most Used
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_file)
    logger.info(f"generate_route_map_by_daytype() - Map saved to {output_file}")

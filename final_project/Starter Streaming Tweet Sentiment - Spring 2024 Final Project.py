# Databricks notebook source
# MAGIC %md
# MAGIC **Submission comment**
# MAGIC
# MAGIC GitHub URL: https://github.com/seant2436/dscc202-402-spring2024.git
# MAGIC
# MAGIC Vocareum Username: labuser104917-3014551@vocareum.com

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSCC202-402 Data Science at Scale Final Project
# MAGIC ### Tracking Tweet sentiment at scale using a pretrained transformer (classifier)
# MAGIC <p>Consider the following illustration of the end to end system that you will be building.  Each student should do their own work.  The project will demonstrate your understanding of Spark Streaming, the medalion data architecture using Delta Lake, Spark Inference at Scale using an MLflow packaged model as well as Exploritory Data Analysis and System Tracking and Monitoring.</p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/pipeline.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You will be pulling an updated copy of the course GitHub repositiory: <a href="https://github.com/lpalum/dscc202-402-spring2024">The Repo</a>.  If you are unclear on how to pull an updated copy using the GitHub command line, the following <a href="https://techwritingmatters.com/how-to-update-your-forked-repository-on-github">document</a> is helpful.  Be sure to add the professors and TAs as collaborators on your project. 
# MAGIC
# MAGIC - lpalum@gmail.com GitHub ID: lpalum
# MAGIC - ajay.anand@rochester.edu GitHub ID: ajayan12
# MAGIC - divyamunot1999@gmail.com GitHub ID: divyamunot
# MAGIC - ylong6@u.Rochester.edu GitHub ID: NinaLong2077
# MAGIC
# MAGIC Once you have updates your fork of the repository you should see the following template project that is resident in the final_project directory.
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/notebooks.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You can then pull your project into the Databrick Workspace using the <a href="https://www.databricks.training/step-by-step/importing-courseware-from-github/index.html">Repos</a> feature.
# MAGIC Each student is expected to submit the URL of their project on GitHub with their code checked in on the main/master branch.  This illustration highlights the branching scheme that you may use to work on your code in steps and then merge your submission into your master branch before submitting.
# MAGIC </p>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/github.drawio.png">
# MAGIC <p>
# MAGIC Work your way through this notebook which will give you the steps required to submit a complete and compliant project.  The following illustration and associated data dictionary specifies the transformations and data that you are to generate for each step in the medallion pipeline.
# MAGIC </p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/dataframes.drawio.png">
# MAGIC
# MAGIC #### Bronze Data - raw ingest
# MAGIC - date - string in the source json
# MAGIC - user - string in the source json
# MAGIC - text - tweet string in the source json
# MAGIC - sentiment - the given sentiment of the text as determined by an unknown model that is provided in the source json
# MAGIC - source_file - the path of the source json file the this row of data was read from
# MAGIC - processing_time - a timestamp of when you read this row from the source json
# MAGIC
# MAGIC #### Silver Data - Bronze Preprocessing
# MAGIC - timestamp - convert date string in the bronze data to a timestamp
# MAGIC - mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# MAGIC - cleaned_text - the bronze text data with the mentions (@username) removed.
# MAGIC - sentiment - the given sentiment that was associated with the text in the bronze table.
# MAGIC
# MAGIC #### Gold Data - Silver Table Inference
# MAGIC - timestamp - the timestamp from the silver data table rows
# MAGIC - mention - the mention from the silver data table rows
# MAGIC - cleaned_text - the cleaned_text from the silver data table rows
# MAGIC - sentiment - the given sentiment from the silver data table rows
# MAGIC - predicted_score - score out of 100 from the Hugging Face Sentiment Transformer
# MAGIC - predicted_sentiment - string representation of the sentiment
# MAGIC - sentiment_id - 0 for negative and 1 for postive associated with the given sentiment
# MAGIC - predicted_sentiment_id - 0 for negative and 1 for positive assocaited with the Hugging Face Sentiment Transformer
# MAGIC
# MAGIC #### Application Data - Gold Table Aggregation
# MAGIC - min_timestamp - the oldest timestamp on a given mention (@username)
# MAGIC - max_timestamp - the newest timestamp on a given mention (@username)
# MAGIC - mention - the user (@username) that this row pertains to.
# MAGIC - negative - total negative tweets directed at this mention (@username)
# MAGIC - neutral - total neutral tweets directed at this mention (@username)
# MAGIC - positive - total positive tweets directed at this mention (@username)
# MAGIC
# MAGIC When you are designing your approach, one of the main decisions that you will need to make is how you are going to orchestrate the streaming data processing in your pipeline.  There are several valid approaches.  First, you may choose to start the bronze_stream and let it complete (read and append all of the source data) before preceeding and starting up the silver_stream.  This approach has latency associated with it but it will allow your code to proceed in a linear fashion and process all the data by the end of your notebook execution.  Another potential approach is to start all the streams and have a "watch" method to determine when the pipeline has processed sufficient or all of the source data before stopping and displaying results.  Both of these approaches are valid and have different implications on how you will trigger your steams and how you will gate the execution of your pipeline.  Think through how you want to proceed and ask questions if you need guidance. The following references may be helpful:
# MAGIC - [Spark Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
# MAGIC - [Databricks Autoloader - Cloudfiles](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC
# MAGIC ### Be sure that your project runs end to end when *Run all* is executued on this notebook! (15 Points out of 60)

# COMMAND ----------

# DBTITLE 1,Pull in the Includes & Utiltites
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 1,Notebook Control Widgets (maybe helpful)
"""
Adding a widget to the notebook to control the clearing of a previous run.
or stopping the active streams using routines defined in the utilities notebook
"""
dbutils.widgets.removeAll()

dbutils.widgets.dropdown("clear_previous_run", "No", ["No","Yes"])
if (getArgument("clear_previous_run") == "Yes"):
    clear_previous_run()
    print("Cleared all previous data.")

dbutils.widgets.dropdown("stop_streams", "No", ["No","Yes"])
if (getArgument("stop_streams") == "Yes"):
    stop_all_streams()
    print("Stopped all active streams.")

from delta import *
dbutils.widgets.dropdown("optimize_tables", "No", ["No","Yes"])
if (getArgument("optimize_tables") == "Yes"):
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, BRONZE_DELTA).optimize().executeCompaction()
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, SILVER_DELTA).optimize().executeCompaction()
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, GOLD_DELTA).optimize().executeCompaction()
    print("Optimized all of the Delta Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.0 Import your libraries here...
# MAGIC - Are your shuffle partitions consistent with your cluster and your workload?
# MAGIC - Do you have the necessary libraries to perform the required operations in the pipeline/application?

# COMMAND ----------

# Importing Spark session and data types
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.functions import (
    col, explode, split, regexp_replace, isnan, when, count, from_json, to_timestamp, 
    current_timestamp, udf
)
from pyspark.sql.types import StringType, FloatType, IntegerType, StructType, StructField, TimestampType
from pyspark.sql.streaming import DataStreamReader

# Importing MLlib for evaluation metrics
from pyspark.mllib.evaluation import MulticlassMetrics

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Importing external libraries
from transformers import pipeline

import mlflow

# COMMAND ----------

# Set the shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "100")  

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

spark.conf.get("spark.sql.adaptive.enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Use the utility functions to ...
# MAGIC - Read the source file directory listing
# MAGIC - Count the source files (how many are there?)
# MAGIC - print the contents of one of the files

# COMMAND ----------

# Create Spark session
spark = (SparkSession
        .builder
        .appName("Twitter Sentiment Analysis")
        .getOrCreate())

schema = StructType([
    StructField("date", StringType(), True),
    StructField("user", StringType(), True),
    StructField("text", StringType(), True),
    StructField("sentiment", StringType(), True),
    StructField("source_file", StringType(), True)
])

tweets_df = (spark
            .readStream 
            .format("cloudFiles") 
            .option("cloudFiles.format", "json") 
            .option("path", "s3a://voc-75-databricks-data/voc_volume/") 
            .schema(schema) 
            .load())

# COMMAND ----------


spark = SparkSession.builder.appName("Tweet Analysis").getOrCreate()

files = dbutils.fs.ls(TWEET_SOURCE_PATH)

num_files = len(files)
print("Number of files:", num_files)

file_contents = dbutils.fs.head(files[0].path)
print(file_contents)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.0 Transform the Raw Data to Bronze Data using a stream
# MAGIC - define the schema for the raw data
# MAGIC - setup a read stream using cloudfiles and the source data format
# MAGIC - setup a write stream using cloudfiles to append to the bronze delta table
# MAGIC - enforce schema
# MAGIC - allow a new schema to be merged into the bronze delta table
# MAGIC - Use the defined BRONZE_CHECKPOINT and BRONZE_DELTA paths defines in the includes
# MAGIC - name your raw to bronze stream as bronze_stream
# MAGIC - transform the raw data to the bronze data using the data definition at the top of the notebook

# COMMAND ----------

# Set Spark to use the legacy time parser policy
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

schema = StructType([
    StructField("date", StringType(), True),
    StructField("user", StringType(), True),
    StructField("text", StringType(), True),
    StructField("sentiment", StringType(), True)
])

# Read the source file directory as a stream using the schema
raw_stream = (spark.readStream
    .schema(schema)
    .option("maxFilesPerTrigger", 1000)  # Process 1000 file per trigger to simulate streaming
    .json(TWEET_SOURCE_PATH))

# Transform the raw stream to add a timestamp and processing_time, and drop the original 'date' column
bronze_data = raw_stream \
    .withColumn("timestamp", to_timestamp(col("date"), "EEE MMM dd HH:mm:ss zzz yyyy")) \
    .withColumn("processing_time", current_timestamp()) \
    .drop("date")

# Write the transformed bronze_data stream out to the Bronze Delta table
bronze_stream = (bronze_data
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", BRONZE_CHECKPOINT)
    .start(BRONZE_DELTA))


# COMMAND ----------

bronze_stream.status

# COMMAND ----------

bronze_table_df = spark.read.format("delta").load(BRONZE_DELTA)
display(bronze_table_df.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.0 Bronze Data Exploratory Data Analysis
# MAGIC - How many tweets are captured in your Bronze Table?
# MAGIC - Are there any columns that contain Nan or Null values?  If so how many and what will you do in your silver transforms to address this?
# MAGIC - Count the number of tweets by each unique user handle and sort the data by descending count.
# MAGIC - How many tweets have at least one mention (@) how many tweet have no mentions (@)
# MAGIC - Plot a bar chart that shows the top 20 tweeters (users)
# MAGIC

# COMMAND ----------

tweet_count = spark.read.format("delta").load(BRONZE_DELTA).count()

display(tweet_count)

# COMMAND ----------

# Load the Bronze Delta table into a DataFrame
bronze_df = spark.read.format("delta").load(BRONZE_DELTA)

# Count Null and NaN values for each column
null_counts = bronze_df.select([count(when(col(c).isNull(), c)).alias(c) for c in bronze_df.columns])
null_counts.show()

# COMMAND ----------

tweets_by_user = spark.read.format("delta").load(BRONZE_DELTA).groupBy("user").count().orderBy(col("count").desc())
display(tweets_by_user)

# COMMAND ----------

tweets_with_mentions = spark.read.format("delta").load(BRONZE_DELTA).filter(col("text").contains("@")).count()
tweets_without_mentions = spark.read.format("delta").load(BRONZE_DELTA).filter(~col("text").contains("@")).count()
display(tweets_with_mentions)

# COMMAND ----------

display(tweets_without_mentions)

# COMMAND ----------

# Top 20 Tweeters Bar Chart
top_tweeters = tweets_by_user.limit(20).toPandas()


plt.figure(figsize=(10, 6))
plt.bar(top_tweeters['user'], top_tweeters['count'], color='skyblue')
plt.xlabel('User Handles')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=90)
plt.title('Top 20 Tweeters')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Transform the Bronze Data to Silver Data using a stream
# MAGIC - setup a read stream on your bronze delta table
# MAGIC - setup a write stream to append to the silver delta table
# MAGIC - Use the defined SILVER_CHECKPOINT and SILVER_DELTA paths in the includes
# MAGIC - name your bronze to silver stream as silver_stream
# MAGIC - transform the bronze data to the silver data using the data definition at the top of the notebook

# COMMAND ----------

# Read the Bronze Delta table as a stream
bronze_df = (
    spark.readStream
    .format("delta")
    .load(BRONZE_DELTA)  
)

def bronze_to_silver(bronze_dataframe):
    # Extract mentions from the text
    mentions_df = bronze_dataframe.withColumn("mentions", explode(split(regexp_replace(col("text"), "[^@\\w]", " "), " ")))
    mentions_df = mentions_df.filter(mentions_df.mentions.startswith("@"))
    
    # Create a cleaned_text column by removing mentions from the text
    cleaned_text_df = mentions_df.withColumn("cleaned_text", regexp_replace(col("text"), "@\\w+", ""))
    
    # Retain the sentiment as is
    silver_data = cleaned_text_df.withColumn("sentiment", col("sentiment"))
    
    return silver_data

# Transform the data
silver_data = bronze_to_silver(bronze_df)

# Write the transformed silver_data stream out to the Silver Delta table
silver_stream = (
    silver_data
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", SILVER_CHECKPOINT)
    .queryName("silver_stream")  
    .start(SILVER_DELTA) 
)

# COMMAND ----------

silver_stream.status

# COMMAND ----------

silver_tweet_count = spark.read.format("delta").load(SILVER_DELTA).count()

display(silver_tweet_count)

# COMMAND ----------

silver_table_df = spark.read.format("delta").load(SILVER_DELTA)
display(silver_table_df.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.0 Transform the Silver Data to Gold Data using a stream
# MAGIC - setup a read stream on your silver delta table
# MAGIC - setup a write stream to append to the gold delta table
# MAGIC - Use the defined GOLD_CHECKPOINT and GOLD_DELTA paths defines in the includes
# MAGIC - name your silver to gold stream as gold_stream
# MAGIC - transform the silver data to the gold data using the data definition at the top of the notebook
# MAGIC - Load the pretrained transformer sentiment classifier from the MODEL_NAME at the production level from the MLflow registry
# MAGIC - Use a spark UDF to parallelize the inference across your silver data

# COMMAND ----------

# Load the pretrained transformer sentiment classifier
sentiment_analysis = pipeline(model=HF_MODEL_NAME, force_download=True)

# Define a UDF to apply sentiment analysis
def sentiment_score(text):
    result = sentiment_analysis(text)
    sentiment_label = result[0]['label']
    sentiment_map = {'POS': 1, 'NEU': 0, 'NEG': 0}  # Mapping sentiment to IDs
    sentiment_id = sentiment_map.get(sentiment_label, -1)  # Default to -1 if label not found
    return float(result[0]['score']), sentiment_label, sentiment_id

sentiment_udf = udf(sentiment_score, returnType=StructType([
    StructField("predicted_score", FloatType(), False),
    StructField("predicted_sentiment", StringType(), False),
    StructField("predicted_sentiment_id", IntegerType(), False)
]))


# Read the Silver Delta table as a stream
silver_df = (
    spark.readStream
    .format("delta")
    .load(SILVER_DELTA)
)

# Define the transformation logic from Silver to Gold
def silver_to_gold(silver_dataframe):
    gold_data = silver_dataframe.withColumn("sentiment_analysis", sentiment_udf(col("cleaned_text")))
    gold_data = gold_data.select(
        col("timestamp"),
        col("mentions"),
        col("cleaned_text"),
        col("sentiment"),
        col("sentiment_analysis.predicted_score").alias("predicted_score"),
        col("sentiment_analysis.predicted_sentiment").alias("predicted_sentiment"),
        col("sentiment_analysis.predicted_sentiment_id").alias("predicted_sentiment_id")
    )
    return gold_data

# Transform the data
gold_data = silver_to_gold(silver_df)

# Write the transformed gold_data stream out to the Gold Delta table
gold_stream = (
    gold_data
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", GOLD_CHECKPOINT)
    .queryName("gold_stream")
    .start(GOLD_DELTA)
)

# COMMAND ----------

gold_stream.status

# COMMAND ----------

gold_tweet_count = spark.read.format("delta").load(GOLD_DELTA).count()

display(gold_tweet_count)

# COMMAND ----------

gold_table_df = spark.read.format("delta").load(GOLD_DELTA)
display(gold_table_df.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0 Capture the accuracy metrics from the gold table in MLflow
# MAGIC Store the following in an MLflow experiment run:
# MAGIC - Store the precision, recall, and F1-score as MLflow metrics
# MAGIC - Store an image of the confusion matrix as an MLflow artifact
# MAGIC - Store the mdoel name and the MLflow version that was used as an MLflow parameters
# MAGIC - Store the version of the Delta Table (input-silver) as an MLflow parameter

# COMMAND ----------

gold_data = spark.read.format("delta").load(GOLD_DELTA)

# A dictionary to map sentiments to numeric values
sentiment_to_id = {'POS': 1, 'NEU': 0, 'NEG': 0}

# Define a UDF to convert sentiment labels to numeric IDs
def sentiment_to_numeric(label):
    return sentiment_to_id.get(label, -1)  # Returns -1 if label is not in the dictionary

# Register UDF with Spark
sentiment_to_numeric_udf = udf(sentiment_to_numeric, IntegerType())

# Add a new column for numeric sentiment labels
gold_data = gold_data.withColumn("sentiment_id", sentiment_to_numeric_udf(col("predicted_sentiment")))

# Filter out rows where predicted_sentiment_id is null
filtered_data = gold_data.filter(col("predicted_sentiment_id").isNotNull())

# Select the predictions and the true labels
rdd = filtered_data.select(col("predicted_sentiment_id").cast("double"), col("sentiment_id").cast("double")).rdd

# Instantiate metrics object
metrics = MulticlassMetrics(rdd)

# Calculate metrics
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1_score = metrics.fMeasure(1.0)

# COMMAND ----------

# Confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# COMMAND ----------

# Start an MLflow run
with mlflow.start_run():
    # Log metrics
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1_score)

    # Log the confusion matrix image as an artifact
    mlflow.log_artifact('confusion_matrix.png', "confusion_matrix.png")

    # Log model name and MLflow version as parameters
    mlflow.log_param("Model Name", HF_MODEL_NAME)
    mlflow.log_param("MLflow Version", mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.0 Application Data Processing and Visualization
# MAGIC - How many mentions are there in the gold data total?
# MAGIC - Count the number of neutral, positive and negative tweets for each mention in new columns
# MAGIC - Capture the total for each mention in a new column
# MAGIC - Sort the mention count totals in descending order
# MAGIC - Plot a bar chart of the top 20 mentions with positive sentiment (the people who are in favor)
# MAGIC - Plot a bar chart of the top 20 mentions with negative sentiment (the people who are the vilians)
# MAGIC
# MAGIC You may want to use the "Loop Application" widget to control whether you repeateded display the latest plots while the data comes in from your streams before moving on to the next section and cleaning up your run.
# MAGIC
# MAGIC *note: A mention is a specific twitter user that has been "mentioned" in a tweet with an @user reference.

# COMMAND ----------

# Load the Gold Delta Table
gold_data = spark.read.format("delta").load(GOLD_DELTA)

# Aggregate data to count sentiments for each mention and capture timestamps
mention_sentiment_count = gold_data.groupBy("mentions").agg(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp"),
    F.count(F.when(F.col("predicted_sentiment") == "NEU", True)).alias("neutral_count"),
    F.count(F.when(F.col("predicted_sentiment") == "POS", True)).alias("positive_count"),
    F.count(F.when(F.col("predicted_sentiment") == "NEG", True)).alias("negative_count"),
    F.count("*").alias("total_mentions")
).orderBy(F.desc("total_mentions"))


# COMMAND ----------

mention_sentiment_count.show(20)

# COMMAND ----------

def plot_top_mentions(dataframe, sentiment_column, title):
    top_mentions = dataframe.orderBy(F.desc(sentiment_column)).limit(20).toPandas()

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.barh(top_mentions['mentions'], top_mentions[sentiment_column], color='blue')
    plt.xlabel('Counts')
    plt.ylabel('Mentions')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to have the largest values on top
    plt.show()


# COMMAND ----------

# Plot the top 20 mentions with positive sentiment
plot_top_mentions(mention_sentiment_count, 'positive_count', 'Top 20 Mentions with Positive Sentiment')

# Plot the top 20 mentions with negative sentiment
plot_top_mentions(mention_sentiment_count, 'negative_count', 'Top 20 Mentions with Negative Sentiment')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.0 Clean up and completion of your pipeline
# MAGIC - using the utilities what streams are running? If any.
# MAGIC - Stop all active streams
# MAGIC - print out the elapsed time of your notebook.

# COMMAND ----------

# ENTER YOUR CODE HERE
# Stop all active streams
for stream in spark.streams.active:
    stream.stop()
    print(f"Stopped stream: {stream.name}")

# COMMAND ----------

import time

END_TIME = time.time()

# Calculate the elapsed time in seconds
elapsed_time = END_TIME - START_TIME

# Convert elapsed time to minutes and hours
elapsed_minutes = elapsed_time / 60
elapsed_hours = elapsed_minutes / 60

# Print the elapsed time in different units
print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.0 How Optimized is your Spark Application (Grad Students Only)
# MAGIC Graduate students (registered for the DSCC-402 section of the course) are required to do this section.  This is a written analysis using the Spark UI (link to screen shots) that support your analysis of your pipelines execution and what is driving its performance.
# MAGIC Recall that Spark Optimization has 5 significant dimensions of considertation:
# MAGIC - Spill: write to executor disk due to lack of memory
# MAGIC - Skew: imbalance in partition size
# MAGIC - Shuffle: network io moving data between executors (wide transforms)
# MAGIC - Storage: inefficiency due to disk storage format (small files, location)
# MAGIC - Serialization: distribution of code segments across the cluster
# MAGIC
# MAGIC Comment on each of the dimentions of performance and how your impelementation is or is not being affected.  Use specific information in the Spark UI to support your description.  
# MAGIC
# MAGIC Note: you can take sreenshots of the Spark UI from your project runs in databricks and then link to those pictures by storing them as a publicly accessible file on your cloud drive (google, one drive, etc.)
# MAGIC
# MAGIC References:
# MAGIC - [Spark UI Reference Reference](https://spark.apache.org/docs/latest/web-ui.html#web-ui)
# MAGIC - [Spark UI Simulator](https://www.databricks.training/spark-ui-simulator/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.0 How Optimized is my Spark Application 
# MAGIC
# MAGIC
# MAGIC Initially, the code faced issues like analytic connection errors, primarily due to the `maxFilesPerTrigger` being set to 1, which caused extremely slow data processing. Adjusting the partition size to 100, enabling adaptive query processing (AQE) by setting `spark.conf.get("spark.sql.adaptive.enabled")`, and increasing `maxFilesPerTrigger` to 1000 largely resolved these issues.
# MAGIC
# MAGIC Upon inspecting the Spark UI, no spill was observed in any of the streaming tasks. Additionally, employing AQE in the initial task settings helped mitigate data skew. According to the "Stages" tab in the Spark UI, the uniform distribution of task durations and data sizes suggests that skew is not present.
# MAGIC
# MAGIC The "Stages" tab also revealed that shuffle read and write metrics are minimal, with sizes ranging from 68 KiB to 500 bytes. This indicates efficient data partitioning, likely enhanced by optimized operations like joins and aggregations.
# MAGIC
# MAGIC Observations from the "Storage" tab suggest potential inefficiencies in caching; a 0% cache hit ratio indicates that the current caching strategy might not be effective, possibly due to the nature of the wide transformations, such as joins, used in the streaming process.
# MAGIC
# MAGIC Regarding serialization, as observed in the "Executor" tab:
# MAGIC 1. **Storage Memory**: An executor is using 519.2 MiB of storage memory out of a total of 2.1 GiB available across all executors. This significant usage includes cached RDDs or DataFrames.
# MAGIC 2. **Disk Used**: This same executor has utilized 517.1 MiB of disk space, some data or intermediate results had to be spilled to disk, potentially due to insufficient memory for caching or storage of all data in memory.
# MAGIC
# MAGIC Given the modest shuffle sizes, the impact of serialization and deserialization overhead on system performance appears minimal.
# MAGIC
# MAGIC - The following screenshots are provided for reference that address each of the 5S:
# MAGIC   - [Executor tab Screenshot](https://drive.google.com/file/d/1kbmzqbjXnbQhA2Hv6FRDZyd2Zd1kDgP1/view?usp=drive_link)
# MAGIC   - [Storage tab Screenshot](https://drive.google.com/file/d/1-8-PHXYShMet1DT3v4R5kayLpXCgUQTF/view?usp=drive_link)
# MAGIC   - [Shuffle Analysis Screenshot](https://drive.google.com/file/d/1tf08nyYPDWbTtGz1DnHCrwiBoF71iyG0/view?usp=drive_link)
# MAGIC
# MAGIC

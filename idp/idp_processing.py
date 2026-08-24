# Databricks notebook source
# MAGIC %md
# MAGIC ##idp processing

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F


# COMMAND ----------

# MAGIC %md
# MAGIC ####paths
# MAGIC

# COMMAND ----------

base_path = '/Volumes/idp/source_system/source_system/'
landing = f"{base_path}/landing/"
processed = f"{base_path}/processed/"


# COMMAND ----------

# MAGIC %md
# MAGIC ####reading files

# COMMAND ----------

files = dbutils.fs.ls(landing)

file = [f for f in files]

if len(file) > 0:
    df = spark.read\
        .format("binaryFile")\
        .load(landing)\
        .withColumn("file_name",F.element_at(F.split(F.col("path"),"/"),-1))

else:
    print("no file found")

    schema = StructType([])
    df = spark.createDataFrame([], schema)


display(len(file))

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.select(
    F.col("path"),
    F.col("file_name"),
    F.col("content")
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####parsing the document

# COMMAND ----------

df = df.withColumn(
    "content",
    F.ai_parse_document(F.col("content"))
)

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####classifying the files based on the content

# COMMAND ----------

df = df.withColumn(
    "document_type",
    F.call_function(
        "ai_classify",
        F.col("content"),
        F.lit('["Invoice","Purchase_order","Receipt","Other"]'),
        F.create_map(F.lit("version"), F.lit("2.1"))
    )
)

display(df)

# COMMAND ----------

# Extract just the classified label from the VARIANT response
df = df.withColumn(
    "document_type_label",
    F.expr("document_type:response[0].value::string")
)



# COMMAND ----------

df = df.drop("document_type")
df = df.withColumnRenamed("document_type_label","document_type")

df = df.select(
    F.col("path"),
    F.col("file_name"),
    F.col("content"),
    F.col("document_type")
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####separating into different dataframes by the file type

# COMMAND ----------

df_invoice = df.filter(
    F.col("document_type") == "Invoice"
)

df_purchase_order = df.filter(
    F.col("document_type") == "Purchase_order"
)

df_receipt = df.filter(
    F.col("document_type") == "Receipt"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ####schema for invoice

# COMMAND ----------

invoice_schema = """
{
  "invoice_number": {
    "type": "string"
  },
  "invoice_date": {
    "type": "string"
  },
  "vendor": {
    "type": "string"
  },
  "po_number": {
    "type": "string"
  },
  "currency": {
    "type": "string"
  },
  "payment_method": {
    "type": "string"
  },
  "total": {
    "type": "string"
  }
}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ####extracting content for invoice

# COMMAND ----------

df_invoice = df_invoice.withColumn(
    "extracted_content",
    F.call_function(
        "ai_extract",
        F.col("content"),
        F.lit(invoice_schema)
    )
)

# COMMAND ----------

df_invoice = df_invoice.select(
    F.col("path"),
    F.col("file_name"),
    F.col("extracted_content"),
    F.col("document_type")
)
display(df_invoice)

# COMMAND ----------

# MAGIC %md
# MAGIC ####creating columns for invoice

# COMMAND ----------

df_invoice = df_invoice.withColumn(
    "vendor",
    F.expr("extracted_content:response.vendor.value::string")
)\
    .withColumn("invoice_number",
                F.expr("extracted_content:response.invoice_number.value::string"))\
    .withColumn("invoice_date",
                F.expr("extracted_content:response.invoice_date.value::string"))\
    .withColumn("purchase_order_number",
                F.expr("extracted_content:response.po_number.value::string"))\
    .withColumn("currency",
                F.expr("extracted_content:response.currency.value::string"))\
    .withColumn("payment_method",
                F.expr("extracted_content:response.payment_method.value::string"))\
    .withColumn("total",
                F.expr("extracted_content:response.total.value::string"))


df_invoice = df_invoice.withColumn(
    "total",
    F.regexp_replace(F.col("total"), "[^0-9.]", "")
)

df_invoice = df_invoice.select(
    F.col("vendor"),
    F.col("invoice_number"),
    F.col("invoice_date"),
    F.col("purchase_order_number"),
    F.col("currency"),
    F.col("payment_method"),
    F.col("total"),
    F.col("file_name")
)


# COMMAND ----------

display(df_invoice)

# COMMAND ----------

# MAGIC %md
# MAGIC ####schema for purchase order

# COMMAND ----------

purchase_order_schema = """
{
    "buyer": {
        "type": "string"
    },
    "vendor": {
        "type": "string"
    },
    "po_number": {
        "type": "string"
    },
    "po_date": {
        "type": "string"
    },
    "ship_date": {
        "type": "string"
    },
    "currency": {
        "type": "string"
    },
    "total": {
        "type": "string"
    }
}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ####extracting content for purchase order
# MAGIC

# COMMAND ----------

df_purchase_order = df_purchase_order.withColumn(
    "extracted_content",
    F.call_function(
        "ai_extract",
        F.col("content"),
        F.lit(purchase_order_schema)
    )
)

# COMMAND ----------

df_purchase_order = df_purchase_order.select(
    F.col("file_name"),
    F.col("extracted_content"),
    F.col("document_type")
)

display(df_purchase_order)

# COMMAND ----------

# MAGIC %md
# MAGIC ####creating columns for purchase order

# COMMAND ----------

df_purchase_order = df_purchase_order.withColumn(
    "buyer",
    F.expr("extracted_content:response.buyer.value::string")
)\
    .withColumn("vendor",
                F.expr("extracted_content:response.vendor.value::string"))\
    .withColumn("purchase_order_number",
                F.expr("extracted_content:response.po_number.value::string"))\
    .withColumn("purchase_order_date",
                F.expr("extracted_content:response.po_date.value::string"))\
    .withColumn("ship_date",
                F.expr("extracted_content:response.ship_date.value::string"))\
    .withColumn("currency",
                F.expr("extracted_content:response.currency.value::string"))\
    .withColumn("total",
                F.expr("extracted_content:response.total.value::string"))


df_purchase_order = df_purchase_order.withColumn(
    "total",
    F.regexp_replace(F.col("total"), "[^0-9.]", "")
)

df_purchase_order = df_purchase_order.select(
    F.col("buyer"),
    F.col("vendor"),
    F.col("purchase_order_number"),
    F.col("purchase_order_date"),
    F.col("ship_date"),
    F.col("currency"),
    F.col("total"),
    F.col("file_name")
)


# COMMAND ----------

display(df_purchase_order)

# COMMAND ----------

# MAGIC %md
# MAGIC ####schema for receipt

# COMMAND ----------

receipt_schema = """
{
    "vendor": {
        "type": "string"
    },
    "receipt_number": {
        "type": "string"
    },
    "date": {
        "type": "string"
    },
    "total": {
        "type": "string"
    }
}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ####extracting content for receipt

# COMMAND ----------

df_receipt = df_receipt.withColumn(
    "extracted_content",
    F.call_function(
        "ai_extract",
        F.col("content"),
        F.lit(receipt_schema)
    )
)

# COMMAND ----------

df_receipt = df_receipt.select(
    F.col("file_name"),
    F.col("extracted_content"),
    F.col("document_type")
)

display(df_receipt)

# COMMAND ----------

# MAGIC %md
# MAGIC creating columns for receipt

# COMMAND ----------

df_receipt = df_receipt.withColumn(
    "vendor",
    F.expr("extracted_content:response.vendor.value::string")
)\
    .withColumn("receipt_number",
                F.expr("extracted_content:response.receipt_number.value::string"))\
    .withColumn("date",
                F.expr("extracted_content:response.date.value::string"))\
    .withColumn("total",
                F.expr("extracted_content:response.total.value::string"))

df_receipt = df_receipt.withColumn(
    "total",
    F.regexp_replace(F.col("total"), "[^0-9.]", "")
)

df_receipt = df_receipt.select(
    F.col("vendor"),
    F.col("receipt_number"),
    F.col("date"),
    F.col("total"),
    F.col("file_name")
)

# COMMAND ----------

display(df_receipt)

# COMMAND ----------

display(df_invoice)

# COMMAND ----------

display(df_purchase_order)

# COMMAND ----------

# MAGIC %md
# MAGIC ####create or merge table for invoice

# COMMAND ----------

silver_invoice="idp.silver.invoice"

if spark.catalog.tableExists(silver_invoice):

    table1 = DeltaTable.forName(spark, silver_invoice)

    table1.alias("target")\
        .merge(df_invoice.alias("source"),
               "target.file_name = source.file_name")\
        .whenMatchedUpdateAll()\
        .whenNotMatchedInsertAll()\
        .execute()
        
    print("mergeed")

else:
    df_invoice.write\
        .mode("overwrite")\
        .format("delta")\
        .option("delta.enableChangeDataFeed","true")\
        .saveAsTable(silver_invoice)

    print("table created")


# COMMAND ----------

# MAGIC %md
# MAGIC ####create or merge table for purchase order

# COMMAND ----------

silver_purchase_order = "idp.silver.purchase_order"

if spark.catalog.tableExists(silver_purchase_order):

    table1 = DeltaTable.forName(spark, silver_purchase_order)

    table1.alias("target")\
        .merge(df_purchase_order.alias("source"),
               "target.file_name = source.file_name")\
        .whenMatchedUpdateAll()\
        .whenNotMatchedInsertAll()\
        .execute()

    print("merged")

else:
    df_purchase_order.write\
        .mode("overwrite")\
        .format("delta")\
        .option("delta.enableChangeDataFeed","true")\
        .saveAsTable(silver_purchase_order)
    
    print("table created")


# COMMAND ----------

# MAGIC %md
# MAGIC ####create  or merge table for receipt

# COMMAND ----------

silver_receipt = "idp.silver.receipt"

if spark.catalog.tableExists(silver_receipt):

    table1 = DeltaTable.forName(spark, silver_receipt)

    table1.alias("target")\
        .merge(df_receipt.alias("source"),
               "target.file_name = source.file_name")\
        .whenMatchedUpdateAll()\
        .whenNotMatchedInsertAll()\
        .execute()
    
    print("merged")

else:
    df_receipt.write\
        .mode("overwrite")\
        .format("delta")\
        .option("delta.enableChangeDataFeed","true")\
        .saveAsTable(silver_receipt)

    print("table created")
    

# COMMAND ----------

# MAGIC %md
# MAGIC ####moving files for landiing to processed

# COMMAND ----------

files = dbutils.fs.ls(landing)

if len(files) > 0:
    for file in files:
        dbutils.fs.mv(file.path, f"{processed}/{file.name}")

    print("files moved successfully")

else:
    print("no files to move")

# COMMAND ----------


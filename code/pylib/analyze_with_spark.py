from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import functions as F

conf = SparkConf()

from pyspark.sql import SparkSession
spark = SparkSession.builder \
                    .master('local') \
                    .appName('muthootSample1') \
                    .config('spark.executor.memory', '5gb') \
                    .config("spark.cores.max", "4") \
                    .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
#sqlContext.read.parquet('test_db_c.sql.parquets/geneTrees.parquet')
gt = spark.read.parquet('test_db_c.sql.parquets/geneTrees.parquet')
#gt = gt.filter( gt['ebl']=
#gt = gt.filter( gt[k]=v for k,v in arg_dict.items if k in gt.columns )

gt.filter((col('ebl')==20) & (col('tout')==20)).cube('ibl','topology').count().orderBy(
        ...: 'ibl','topology').show()

seq = spark.read.parquet('test_db_c.sql.parquets/seqs.parquet')

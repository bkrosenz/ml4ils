IPYTHON=1 pyspark  --num-executors 5 --driver-memory 2g --executor-memory 2g \
         --conf spark.executor.extraClassPath=sqlite-jdbc-3.23.1.jar --driver-class-path sqlite-jdbc-3.23.1.jar --jars sqlite-jdbc-3.23.1.jar # --master <master-URL>


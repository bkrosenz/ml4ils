import sqlite3

dbfile='/N/dc2/projects/bkrosenz/deep_ils/databases/sim_db.sql'
db=sqlite3.connect(dbfile,check_same_thread=False,
                   isolation_level = None)

db.close()   

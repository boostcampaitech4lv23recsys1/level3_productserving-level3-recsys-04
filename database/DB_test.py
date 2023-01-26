import pandas as pd
import sqlite3

cnxn = sqlite3.connect("a.db")
cursor = cnxn.cursor()
select_sql = "select * from test1"  # where rating = 4.42"
cursor.execute(select_sql)
result = cursor.fetchall()
print(result)

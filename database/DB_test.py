import pymysql
cnxn = pymysql.connect(host='1.238.57.88', user='newuser', password='ghdwogud1028', charset='utf8', port = 9876)
# cnxn = pymysql.connect(host='1.238.57.88', user='newuser', password='ghdwogud1028', db='newproj', charset='utf8', port = 9876)
cursor = cnxn.cursor()

import pymysql
cnxn = pymysql.connect(host='1.238.57.88', user='newuser', password='ghdwogud1028', charset='utf8', port = 9876)
# cnxn = pymysql.connect(host='1.238.57.88', user='newuser', password='ghdwogud1028', db='newproj', charset='utf8', port = 9876)
cursor = cnxn.cursor()
dirty = [
    0,  1,  2,
    3,  4,  5,
    6,  7,  8,
]



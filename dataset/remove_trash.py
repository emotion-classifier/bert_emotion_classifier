import csv

f1 = open("tweetlist(park).csv", "r", encoding="utf-8")
rdr = list(csv.reader(f1))
for x in rdr[:]:
    if x[1] == "버림":
        rdr.remove(x)
f1.close()

f1 = open("tweetlist(park).csv", "w", encoding="utf-8")
wr = csv.writer(f1)
wr.writerows(rdr)
f1.close()

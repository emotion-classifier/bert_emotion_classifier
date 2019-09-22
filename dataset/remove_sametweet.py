import csv
f1 = open("same_tweetlist.csv", "r", encoding="utf-8")
rdr = list(csv.reader(f1))
for x in range(len(rdr)):
    rdr[x] = tuple(rdr[x])
print(rdr)
rdr = list(set(rdr))
f1.close()

f1 = open("same_tweetlist.csv", "w", encoding="utf-8")
wr = csv.writer(f1)
wr.writerows(rdr)
f1.close()

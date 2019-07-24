import csv

f1 = open("tweetlist(lim-14816)(1).csv", 'r', encoding="utf-8", newline='')
f2 = open("tweetlist(lim-14816)(2).csv", 'r', encoding="utf-8", newline='')
f3 = open("tweetlist(lim-14816)(same).csv", 'w', encoding='utf-8', newline='')
f4 = open("tweetlist(lim-14816)(diff).csv", 'w', encoding='utf-8', newline='')

twitlist1 = list()
twitlist2 = list()

rdr1 = csv.reader(f2)
count = 0

for line in rdr1:
    count += 1
print(count)

f2.close()
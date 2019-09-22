import csv
f = open("same_tweetlist.csv", "r", encoding="utf-8")
rdr = list(csv.reader(f))
f1 = open("only_happy.csv", 'w', encoding="utf-8")
f2 = open("only_sad.csv", 'w', encoding="utf-8")
f3 = open("only_upset.csv", 'w', encoding="utf-8")
f4 = open("only_444.csv", 'w', encoding="utf-8")
f5 = open("only_neutral.csv", 'w', encoding="utf-8")
only_happy = csv.writer(f1)
only_sad = csv.writer(f2)
only_upset = csv.writer(f3)
only_444 = csv.writer(f4)
only_neutral = csv.writer(f5)

for x in rdr:
    if x[1] == "기쁨":
        only_happy.writerow(x)
    elif x[1] == "슬픔":
        only_sad.writerow(x)
    elif x[1] == "화남":
        only_upset.writerow(x)
    elif x[1] == "불안":
        only_444.writerow(x)
    elif x[1] == "중립":
        only_neutral.writerow(x)
    else:
        print(x)
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
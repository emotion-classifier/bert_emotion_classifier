import csv
f1 = open("only_happy.csv", 'r', encoding="utf-8")
f2 = open("only_sad_add_from_nsmc.csv", 'r', encoding="utf-8")
f3 = open("only_upset.csv", 'r', encoding="utf-8")
f4 = open("only_444.csv", 'r', encoding="utf-8")
f5 = open("only_neutral.csv", 'r', encoding="utf-8")
happy = list(csv.reader(f1))
sad = list(csv.reader(f2))
upset = list(csv.reader(f3))
only_444 = list(csv.reader(f4))
neutral = list(csv.reader(f5))

sentiment_list = [happy, sad, upset, neutral]

train_file = open("balanced_add_sad_train_data.csv", "w", encoding="utf-8")
test_file = open("balanced_add_sad_test_data.csv", 'w', encoding="utf-8")
tr_wr = csv.writer(train_file)
te_wr = csv.writer(test_file)

for s in sentiment_list:
    """if s is neutral:
        tr_wr.writerows(s[:int(len(s) / 2 * 0.9)])
        te_wr.writerows(s[int(len(s) / 2 * 0.9):])
    else:"""
    tr_wr.writerows(s[:int(len(s)*0.9)])
    te_wr.writerows(s[int(len(s)*0.9):])

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
train_file.close()
test_file.close()
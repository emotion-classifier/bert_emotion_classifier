import csv

f1 = open("tokened_traindataset.csv", 'r', encoding="utf-8")
rdr = csv.reader(f1)
data_list = list()
for x in rdr:
    if len(x) == 0:
        continue
    elif len(x) > 2:
        label = x[-1]
        data = ""
        for y in range(len(x)-1):
            data += x[y]
        data_list.append((data, label))
    elif len(x) == 2:
        data_list.append(x)
    else:
        print(x)
f1.close()
f2 = open("train_temp.csv", "w", encoding="utf-8")
wr = csv.writer(f2)
wr.writerows(data_list)
f2.close()
f2 = open("train_temp.csv", "r", encoding="utf-8")
data_list = f2.read()
for x in range(20000):
    data_list = data_list.replace("\n\n", "\n")
f2.close()
f1 = open("tokened_traindataset.csv", 'w', encoding="utf-8")
f1.write(data_list)
f1.close()
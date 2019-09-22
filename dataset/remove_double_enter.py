namelist = ["only_444.csv", "only_happy.csv", "only_neutral.csv", "only_sad.csv", "only_upset.csv"]
for fname in namelist:
    f2 = open(fname, "r", encoding="utf-8")
    data_list = f2.read()
    for x in range(20000):
        data_list = data_list.replace("\n\n", "\n")
    f2.close()
    f1 = open(fname, 'w', encoding="utf-8")
    f1.write(data_list)
    f1.close()

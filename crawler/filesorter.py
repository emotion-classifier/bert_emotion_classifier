f = open("names_", "r", encoding="utf-8")
f2 = open("su", "w", encoding = "utf-8")

raw_data = f.read().split("\n")
for x in raw_data:
    raw_data.remove(x)
    raw_data.append(x.strip())

data = list()
for x in raw_data:
    if len(x) == 3:
        data.append(x)
data = sorted(data)[:int(len(data)/2)]

for x in data:
    f2.write(x+"\n")
f.close()
f2.close()

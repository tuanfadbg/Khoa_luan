import csv

r = csv.reader(open('output_covid.csv'))
lines = list(r)

r2 = csv.reader(open('output_none.csv'))
lines2 = list(r2)

train = []
test = []

ratio = 0.8

for i in range(1, int(len(lines) * ratio)):
    train.append(lines[i])

for i in range(int(len(lines) * ratio) + 1, len(lines)):
    test.append(lines[i])

for i in range(1, int(len(lines2) * ratio)):
    train.append(lines2[i])

for i in range(int(len(lines2) * ratio) + 1, len(lines2)):
    test.append(lines2[i])

writer_train = csv.writer(open('split/train.csv', 'w'))
writer_train.writerows(train)

writer_test = csv.writer(open('split/test.csv', 'w'))
writer_test.writerows(test)

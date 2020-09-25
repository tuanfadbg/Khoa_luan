import csv

r = csv.reader(open('output_covid.csv'))
lines = list(r)

r2 = csv.reader(open('output_none.csv'))
lines2 = list(r2)

output = []

for i in range(1, len(lines)):
    output.append(lines[i])

for i in range(1, len(lines2)):
    output.append(lines2[i])

writer_train = csv.writer(open('combine/output.csv', 'w'))
writer_train.writerows(output)

import csv

r = csv.reader(open('output_temp.csv'))
lines = list(r)

r2 = csv.reader(open('data_new_non_covid.csv'))
lines2 = list(r2)

count = 0
title = []
for i in range(len(lines)):
    title.append(lines[i][14].lower())
    # print(title)

for i2 in range(len(lines2)):
    title2 = lines2[i2][1]
    # print(title2)
    if title2 in title:
        count += 1
        # lines.append(lines2[i2])
    else:
        item = []
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append('')
        item.append(lines2[i2][1])
        item.append(lines2[i2][2])
        item.append(lines2[i2][3])
        item.append(lines2[i2][4])
        item.append(lines2[i2][5])
        lines.append(item)

print(str(count) + " of " + str(len(lines)))
writer = csv.writer(open('output_temp_2.csv', 'w'))
writer.writerows(lines)

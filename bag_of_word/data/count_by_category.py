import csv

r = csv.reader(open('output_none.csv'))
lines = list(r)

categories = {}

for i in range(len(lines) - 1):
    item = lines[i + 1]

    # title = item[1]
    # des = item[2]
    # content = item[3]
    # keyword = item[4]
    category = item[17]
    category_arr = category.split('|')

    if category_arr[0] in categories:
        categories[category_arr[0]] += 1
    else:
        categories[category_arr[0]] = 1

    if len(category_arr) > 1:
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1

print(categories)

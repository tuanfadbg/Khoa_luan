import csv

from pyvi import ViTokenizer


class Segmentation(object):
    def __init__(self):
        self.loop_count = 5
        is_covid_position = 13
        title_position = 14
        des_position = title_position + 1
        content_position = des_position + 1
        category_position = content_position + 1
        keyword_position = category_position + 1

        r = csv.reader(open('../data/combine/output.csv'))
        self.data_csv = list(r)
        self.data = []
        self.is_covid = []

        self.loop_count = len(self.data_csv)

        for i in range(self.loop_count):
            str = self.data_csv[i][title_position] + " " \
                  + self.data_csv[i][des_position] + " " \
                  + self.data_csv[i][content_position] + " " \
                  + self.data_csv[i][category_position] + " " \
                  + self.data_csv[i][keyword_position]
            self.data.append(str)

            self.is_covid.append(self.data_csv[i][is_covid_position])

    def segmentation(self):
        for i in range(self.loop_count):
            self.data[i] = ViTokenizer.tokenize(self.data[i])
            # print(self.data[i])

    def save_data(self):
        # employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # print(self.data)
        save_data = []

        for i in range(self.loop_count):
            save_data.append([self.is_covid[i], self.data[i]])
        # writer_test = csv.writer(open('data_segmentation.csv', 'w'), delimiter='[', quotechar='', quoting=csv.QUOTE_MINIMAL)
        writer_test = csv.writer(open('data_segmentation.csv', 'w'))
        writer_test.writerows(save_data)


dic = Segmentation()
dic.segmentation()
# dic.create_dictionary()
dic.save_data()

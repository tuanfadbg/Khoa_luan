import scrapy

class BaoMoiSpider(scrapy.Spider):
    name = "CrawlerBaoMoiFastest"
    start_urls = [
        'https://baomoi.com/chu-de.epi',
    ]
    base_url = 'https://baomoi.com/ignored/c/'
    count = 0

    # def request_new_post(self, post_id):
    #
    #     pass

    # def start_requests(self):
    #     cars = []
    #     for x in range(20):
    #         cars.append("https://baomoi.com/tin-moi/trang" + str(x + 1) + ".epi")
    #     print(cars)
    #     return cars

    def parse(self, response):
        if self.count > 300000:
            return

        body_arr = response.xpath('//*[@class="body-text"]//text()').extract()
        body_arr = map(lambda x: x.strip(), body_arr)

        keyword_arr = response.xpath('//*[@class="keyword"]//text()').extract()
        keyword_arr = map(lambda x: x.strip(), keyword_arr)

        category_arr = response.xpath('//*[@class="cate"]//text()').extract()
        category_arr = map(lambda x: x.strip(), category_arr)

        title = response.css('h1.article__header::text').get()
        des = response.css('div.article__sapo::text').get()
        content = ' '.join(body_arr)

        if title and content:
            self.count += 1


            yield {
                'title': title,
                'des': des,
                'content': content,
                'keyword': '|'.join(keyword_arr),
                'category': '|'.join(category_arr)
            }

        story_link_arr = response.xpath('//*[@class="story"]/@data-aid').extract()
        story_link_arr = map(lambda x: x.strip(), story_link_arr)
        for post_id in story_link_arr:
            # self.request_new_post(self, post_id)
            yield scrapy.Request(self.base_url + post_id + '.epi', callback=self.parse)

        relate_link_arr = response.xpath('//*[@class="relate"]/@href').extract()
        relate_link_arr = map(lambda x: x.strip(), relate_link_arr)
        for post_id in relate_link_arr:
            # self.request_new_post(self, post_id)
            link_id = post_id[0:-4]
            yield scrapy.Request("https://baomoi.com" + post_id, callback=self.parse)
            # for x in range(1):
            # yield scrapy.Request("https://baomoi.com" + link_id + "/trang" + str(0 + 2) + ".epi",
            #                      callback=self.parse)

        keyword_link_arr = response.xpath('//*[@class="keyword"]/@href').extract()
        keyword_link_arr = map(lambda x: x.strip(), keyword_link_arr)
        for post_id in keyword_link_arr:
            # self.request_new_post(self, post_id)
            link_id = post_id[0:-4]
            yield scrapy.Request("https://baomoi.com" + post_id, callback=self.parse)
            # for x in range(1):
            # yield scrapy.Request("https://baomoi.com" + link_id + "/trang" + str(0 + 2) + ".epi",
            #                   callback=self.parse)

# Go to /spiders and run in command
# scrapy crawl CrawlerBaoMoiFastest -o test_model_1.csv

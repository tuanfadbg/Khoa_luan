import csv
import pickle
import random
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from bag_of_word.settings import STOP_WORDS
from bag_of_word.split_word import SplitWord

start_time = time.time()
folder = "model"
r = csv.reader(open('../data/combine/output_temp_2.csv'))
data_csv = list(r)

loop_count = len(data_csv)
# loop_count = 16000
data = []
is_covid = np.empty([loop_count, ])

sample_items = []
if loop_count == len(data_csv):
    sample_items = data_csv
else:
    sample_items = random.sample(data_csv, loop_count)

stopwords = []
with open(STOP_WORDS, 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
offset = 13
for i in range(loop_count):
    # data.append(sample_items[i][1])
    str = sample_items[i][offset + 1] + ' ' + sample_items[i][offset + 2] + ' ' + sample_items[i][offset + 3] + ' ' + sample_items[i][offset + 4] + ' ' + sample_items[i][offset + 5]
    # if i == 1:
    #     print(str)
    data.append(SplitWord(str, stopwords).segmentation_remove_stop_word())
    if sample_items[i][offset]:
        is_covid[i] = 1
    else:
        is_covid[i] = 0

    # if i < 10:
    #     print(sample_items[i][offset])
    #     print(is_covid[i])

# cv = CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)
cv = CountVectorizer()

word_count_vector = cv.fit_transform(data)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
pickle.dump(tfidf_transformer, open(folder + "/logistic_regression_tfidf_transformer.pkl", 'wb'))
pickle.dump(cv, open(folder + "/logistic_regression_count_vectorizer.pkl", 'wb'))

count_vector = cv.transform(data)

tf_idf_vector = tfidf_transformer.transform(count_vector)

feature_names = cv.get_feature_names()
X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, is_covid, test_size=0.3, random_state=42)

text_clf = LogisticRegression(solver='lbfgs',
                              multi_class='auto',
                              max_iter=10000)
text_clf = text_clf.fit(X_train, y_train)

train_time = time.time() - start_time
print('Done training Logistic regression in', train_time, 'seconds.')

print(text_clf.score(X_test, y_test))
print(classification_report(X_test, y_test))

# Save model
pickle.dump(text_clf, open(folder + "/logistic_regression.pkl", 'wb'))
test = ["Dịch bệnh covid 19",
        "Dịch bệnh covid 19 Dịch bệnh covid 19 Dịch bệnh covid 19 Dịch bệnh covid 19 Việt Nam là một trong số ít nước được IMF dự báo có thể tăng trưởng dương với GDP cuối năm đạt hơn 340 tỷ USD, vượt Singapore. Con số này tăng so với năm ngoái, giúp quy mô nền kinh tế Việt Nam vượt Singapore (337 tỷ USD) và Malaysia (336 tỷ USD). Quỹ Tiền tệ quốc tế (IMF) cũng dự báo GDP bình quân đầu người của Việt Nam tăng từ 3.416 USD năm ngoái lên gần 3.500 USD năm nay",
        "Dịch bệnh ‘bào mòn’ nửa lợi nhuận của các tập đoàn nhà nước. Do ảnh hưởng của dịch bệnh, lợi nhuận của 55 Tập đoàn, Tổng công ty nhà nước năm nay dự báo giảm hơn 45% so với năm 2019. Theo báo cáo của Chính phủ gửi Quốc hội liên quan đến quản lý sử dụng vốn nhà nước tại doanh nghiệp, do ảnh hưởng của Covid-19 nên hầu hết tập đoàn, Tổng công ty nhà nước có thể không đạt mục tiêu lợi nhuận năm 2020. Việc này đã được cơ quan đại diện chủ sở hữu phê duyệt. Số liệu tổng hợp từ 55 Tập đoàn, Tổng công ty nhà nước cho thấy, tổng doanh thu năm nay dự kiến hơn 1,3 triệu tỷ đồng, giảm gần 10% so với năm 2019. Theo đó, lãi trước thuế dự kiến giảm hơn 45% xuống còn 78.000 tỷ. Báo cáo của Chính phủ cũng cho biết, nhiều doanh nghiệp nhà nước và có vốn góp nhà nước thua lỗ lớn trong 6 tháng đầu năm và dự báo còn gặp nhiều khó khăn, ảnh hưởng đến số thu ngân sách trong năm nay và các năm tới. Trong 6 tháng đầu năm, Tập đoàn Hóa chất Việt Nam lỗ hơn 1.000 tỷ đồng, dự kiến quý III tiếp tục lỗ hơn 500 tỷ đồng (năm 2019 lỗ 1.595 tỷ đồng). Tập đoàn Xăng dầu Việt Nam 6 tháng đầu năm lỗ khoảng 220 tỷ đồng... Bên cạnh đó, Tổng công ty Hàng không Việt Nam dự kiến lỗ năm 2020 khoảng 15.000-16.000 tỷ đồng và phải tới năm 2024 mới hết lỗ.",
        "nCovi, covid-19 au một năm thấp kỷ lục, giá cá tra nguyên liệu miền Tây tăng trở lại nhưng người nuôi đã đuối sức vì thua lỗ. Chủ một doanh nghiệp chế biến thuỷ sản lớn ở Cần Thơ cho biết, hai tuần qua giá cá tra thị trường tiêu thụ khả quan hơn khi Trung Quốc bắt đầu tiêu thụ mạnh trở lại. Thị trường Mỹ, kênh tiêu thụ nhà hàng giảm nhưng chuỗi siêu thị tăng nhu cầu gấp bốn lần, cùng với các thị trường khác như EU, Đông Nam Á cũng bắt đầu tăng nhu cầu chuẩn bị nguồn hàng cho dịp cuối năm. Giá cá tra nguyên liệu các doanh nghiệp mua vào lên 21.000-22.000 đồng mỗi kg Theo nhà chế biến xuất khẩu này, nếu người nuôi chuyên nghiệp đúng lúc giá cá giống thấp (khoảng 20.000 đồng mỗi kg loại 30 con), giá thành cũng khoảng 18.000-19.000 đồng mỗi kg. Với giá bán hiện tại là có lời, chủ doanh nghiệp nói Tuy nhiên, các trường hợp thua lỗ rơi vào những người nuôi lúc giá cá giống rất cao (50.000-70.000 đồng mỗi kg) và kéo dài khiến giá thành tăng.",
        "Nguồn tin của Reuters cho biết Bộ Ngoại giao Mỹ đã trình đề xuất lên chính phủ về việc bổ sung Ant Group vào danh sách đen về thương mại. Việc này diễn ra trước thềm IPO của đại gia công nghệ tài chính Trung Quốc. Ant Group sẽ niêm yết trên hai sàn Hong Kong và Thượng Hải, dự kiến huy động số tiền kỷ lục 35 tỷ USD Những quan chức Mỹ có quan điểm cứng rắn với Trung Quốc đang tìm cách ngăn nhà đầu tư Mỹ tham gia vào IPO này. Chính quyền Tổng thống Mỹ Donald Trump lo ngại Ant có thể cung cấp dữ liệu cá nhân quan trọng của người dùng Mỹ cho chính phủ Trung Quốc. Hiện chưa rõ khi nào chính phủ Mỹ sẽ xem xét vấn đề danh sách đen.",
        "Thu nhập khoảng 10 triệu đồng, nhưng có tháng bác Tuấn phải sang vay tạm bố mẹ tôi vài trăm ngàn để đi ăn cỗ Bác Tuấn là trung tá bộ đội về hưu, lương hưu của bác thuộc hàng cao gần nhất xã tôi. Cộng với vài khoản khác như phụ cấp từ vị trí chủ tịch hội cựu chiến binh xã, thành viên ban chấp hành hội nông dân huyện, nhưng chẳng mấy khi bác thảnh thơi chuyện tiền bạc. Lâu lâu bác chạy sang mượn tạm bố tôi một ít vì chưa đến ngày nhận lương Đó là điều khó tin. Bởi với thu nhập trên, ở nông thôn, vợ chồng bác phải sống rất dư dật mới đúng. Ngoài ra, với bản tính siêng năng, vườn nhà bác lúc nào cũng có rau xanh, hai bác còn nuôi được khá nhiều gà, vịt, tự cung tự cấp và biếu hàng xóm. Vì là chỗ thân tình, có lần bác chân thành chia sẻ với tôi về cơ cấu chi tiêu. Hàng tháng bác đang hỗ trợ vài triệu đồng cho đứa cháu nội học đại học ở Hà Nội, cộng thêm cô con gái út mới đi nước ngoài xuất khẩu lao động, phải vay ngân hàng một ít lo chi phí nên thêm khoản lãi hơn triệu đồng mỗi tháng. Còn lại, thu nhập của hai bác chỉ tập trung chi tiêu trong gia đình, mà nặng nhất là khoản đám đình, cưới hỏi, ma chay và các hữu sự khác trong vùng.",
        "Trump nói con út Barron 'không biết mình nhiễm nCoV Trump nói Barron đánh bại Covid-19 trong thời gian rất ngắn, cho rằng đây là lý do nên mở lại trường học càng sớm càng tốt. Trong chuyến vận động tranh cử tại Des Moines, thủ phủ bang Iowa hôm 14/10, Tổng thống Mỹ Donald Trump đã nhắc tới con trai Barron như một lý do để các trường học Mỹ nên mở cửa lại. Trước đó, đệ nhất phu nhân Melania Trump tiết lộ Barron cũng nhiễm nCoV nhưng chỉ xuất hiện triệu chứng nhẹ và đã sạch virus. Tôi thậm chí không hề nghĩ rằng thằng bé biết mình bị nhiễm, Trump nói về Barron. Bởi chúng còn trẻ, hệ miễn dịch rất mạnh và đánh bại virus 99,9%. Barron giờ rất khỏe và đã sạch virus. Barron đã xét nghiệm dương tính. Trong thời gian ngắn, thằng bé giờ đã ổn. Nó lại xét nghiệm âm tính. Người ta nhiễm rồi tự khỏi. Do đó, hãy đưa bọn trẻ trở lại trường học, chúng ta phải cho trẻ con đi học lại, Trump tuyên bố. Tổng thống Mỹ Donald Trump cùng đệ nhất phu nhân Melania, con trai út Barron, rời trực thăng hướng về máy bay Không lực một tại sân bay thành phố Morristown, bang New Jersey, hôm 16/8. Ảnh: AP Tổng thống Mỹ Donald Trump cùng đệ nhất phu nhân Melania, con trai út Barron, rời trực thăng hướng về chuyên cơ Không lực Một tại sân bay thành phố Morristown, bang New Jersey, hôm 16/8. Ảnh: AP. Tổng thống Mỹ đã tìm cách thuyết phục các bang mở cửa lại trường học và nền kinh tế, nhưng các liên đoàn giáo viên phản đối, cho rằng giáo viên cao tuổi có thể bị lây nhiễm nCoV từ học sinh. Cuộc mít tinh của Trump tại Iowa là một phần trong chuyến vận động tranh cử tại những bang quan trọng để quyết định ai sẽ giành chiến thắng trong cuộc bầu cử ngày 3/11. 20 ngày nữa kể từ hôm nay, chúng ta sẽ giành chiến thắng ở bang này, Trump nói trong một đêm lộng gió ở Des Moines, nơi nhiều người trong đám đông dự sự kiện không đeo khẩu trang để phòng ngừa loại virus đã khiến gần 216.000 người Mỹ thiệt mạng. Chúng ta sẽ giành thêm 4 năm nữa ở Nhà Trắng. Trump dự kiến tới Bắc Carolina vào 15/10, sau đó vận động ở Florida và Georgia vào 16/10 rồi sẽ tới Michigan và Wisconsin phát biểu vào hôm sau. Đây đều là những bang ông đã giành chiến thắng năm 2016 nhưng có nguy cơ thất thế trước ứng viên đảng Dân chủ Joe Biden năm nay. Trump sẽ vận động tranh cử ở Las Vegas, Nevada vào tối 17/10, sau đó bắt đầu hành trình khác ở các bang phía tây.",
        "Croatia tiếp tục là đối thủ ưa thích của Pháp. Chiến thắng tối 14/10 giúp Gà trống Gaulois nâng số trận bất bại trước đội bóng vùng Balkan lên tám trận, đồng thời chia hai vị trí đầu bảng A3 với Bồ Đào Nha. Trong lịch sử, Croatia rất có duyên với Pháp. World Cup 1998, Didier Deschamps cùng đồng đội thắng ngược đối thủ 2-1 ở bán kết trước khi lần đầu lên đỉnh thế giới. 20 năm sau, khi là HLV, Deschamps tiếp tục giúp Pháp thắng 4-2 và lần thứ hai vô địch thế giới.Tại lượt đi Nations League mùa này, đội bóng áo lam cũng thắng 4-2 lượt đi. Dù được tiếng nói lịch sử ủng hộ, Pháp gặp khó trước lối pressing rát của Croatia ở trận tái đấu. Việc tiếp tục xếp Antonie Griezmann đá lùi xuống sát hàng tiền vệ, Deschamps chưa thể kiểm soát tuyến giữa đúng ý bởi sự xuất sắc của Luka Modric phía bên kia",
        "Thái_Lan tạm dừng hợp_đồng mua 2 tàu_ngầm Trung_Quốc Chính_phủ Thái_Lan đã tạm dừng dự_án mua hai tàu_ngầm đang gây nhiều tranh_cãi trị_giá 22,5 tỷ bạt từ Trung_Quốc . Hôm_nay ( 31 / 8 ) , Bangkok_Post đưa tin , Chính_phủ Thái_Lan đã tạm dừng dự_án mua hai tàu_ngầm đang gây nhiều tranh_cãi trị_giá 22,5 tỷ bạt ( tương_đương khoảng 720 triệu USD ) từ Trung_Quốc sau khi Chính_phủ Trung_Quốc chấp_thuận việc trì_hoãn hợp_đồng mua_bán trong vòng một năm . Ủy_ban Giám_sát ngân_sách tài khóa 2021 của Hạ_viện đã được Chính_phủ Thái_Lan thông_báo về việc Chính_phủ Trung_Quốc chấp_thuận đề_nghị hoãn triển_khai hợp_đồng trên và đề_nghị cắt_giảm dự_toán số tiền trong gói thanh_toán đầu_tiên . Được biết , hợp_đồng mua 3 tàu_ngầm lớp Yuan Trung_Quốc của Hải_quân Thái_Lan được ưu_đãi “ Mua 3 tàu tính giá 2 chiếc ” với tổng ngân_sách được công_bố là 36 tỷ bạt ( tương_đương khoản 1,14 tỷ USD ) , khoản tiền trị_giá 13,5 tỷ bạt ( khoảng 429 triệu USD ) đã được chi cho chiếc tàu đầu_tiên . Ngày 21 / 8 vừa_qua , Tiểu_ban phụ_trách mua_sắm tài_sản công thuộc Hạ_viện Thái_Lan đã bỏ_phiếu thông_qua khoản chi 22,5 tỷ bạt ( tương_đương khoảng 713 triệu USD ) trong năm tài khóa 2021 để thực_hiện hợp_đồng mua chiếc 2 tàu_ngầm lớp Yuan còn lại của Trung_Quốc . Trong bối_cảnh nền kinh_tế Thái_Lan chịu thiệt_hại nặng_nề vì đại_dịch Covid - 19 , nhiều Hạ_nghị_sĩ đối_lập đã phản_đối gay_gắt kế_hoạch này của Hải_quân Thái_Lan . / . Kinh_tế | Tài_chính Tàu_ngầm | Chính_phủ Thái_Lan | Yuan Trung_Quốc | Thái_Lan | Trung_Quốc | Bangkok_Post | Trị_giá | Dự_án | Hạ_viện Thái_Lan | Hợp_đồng | Tranh_cãi | Chính_phủ Trung_Quốc | Hải_quân Thái_Lan | SINA | Hạ_nghị_viện | Trì_hoãn | Tổng ngân_sách | Chấp_thuận | Hoãn | Tương_đương",
        "Khai thác đất trái phép, 'hung thần' xe ben lao thẳng vào công an Khi lực lượng chức năng phát hiện, yêu cầu đoàn xe ben khai thác đất trái phép dừng lại để kiểm tra, tài xế không chấp hành mà lao thẳng vào công an. Cơ quan CSĐT Công an huyện Bắc Tân Uyên (Bình Dương) sáng nay cho biết đang tạm giữ hình sự Đoàn Lê Thanh Tú (34 tuổi, ngụ huyện Bắc Tân Uyên, là tài xế xe ben) để điều tra hành vi “Chống người thi hành công vụ”. Theo điều tra, rạng sáng ngày 13/8, tổ tuần tra của Công an huyện Bắc Tân Uyên làm nhiệm vụ tuần tra tại khu vực xã Tân Mỹ thì phát hiện đoàn xe ben có dấu hiệu khai thác, vận chuyển đất trái phép. Lúc này, tổ tuần tra ra hiệu lệnh dừng xe để kiểm tra thì cả 4 xe ben tăng ga bỏ chạy. Trong số này, tài xế Đoàn Lê Thanh Tú điều khiển xe ben biển số 61C - 6524 tông thẳng vào lực lượng chức năng. Bị tông bất ngờ, lực lượng chức năng tri hô nhảy ra ngoài để tránh nên không ai bị thương, tuy nhiên một xe máy chuyên dụng của tổ tuần tra bị tông văng ra giữa đường. Ngay sau đó, tài xế Tú đã bị Công an khống chế, bắt giữ. Công an huyện Bắc Tân Uyên xác định, khu vực khai thác đất trên không được cấp phép; đồng thời cũng đang điều tra hành vi “Vận chuyển khoáng sản trái phép” đối với 3 tài xế khác trong đoàn xe ben nói trên. Huyện Bắc Tân Uyên là khu vực tập trung nhiều mỏ khoáng sản của tỉnh Bình Dương, nhiều năm nay xảy ra tình trạng khai thác đất đá ồ ạt, xe ben và xe trọng tải hoành hành khiến người dân bức xúc. Pháp luật An ninh - Trật tự Xe ben Đoàn Lê Thanh Tú Hung thần Tân Mỹ Bắc Tân Uyên Trái phép Khai thác Tông Bình Dương Tài xế Chống người thi hành công vụ Đất Công an Chấp hành Tạm giữ Chức năng Dừng lại Vận chuyển Hiệu lệnh Lực lượng",
        "Cây phượng đổ đè người đi đường, 2 người đi cấp cứu Cây phượng đổ đè 3 người đi đường, trong đó 2 người bị thương phải đi cấp cứu vào chiều 31-8. Vào 16 giờ ngày 31-8, một cây phượng lớn bên đường Nguyễn Văn Trỗi (thuộc địa bàn phường Thanh Sơn, TP Phan Rang – Tháp Chàm, tỉnh Ninh Thuận) bất ngờ ngã đổ làm bị thương 2 người đi đường. 2 người bị thương phải đi cấp cứu tại Bệnh viện Đa khoa tỉnh Ninh Thuận. Cây phượng lớn ngã chắn lối đi. Để tránh ùn tắc giao thông, người dân đã nhanh chóng chung tay dọn dẹp. Công an và ngành chức năng cũng kịp thời có mặt để giải quyết sự việc. Nguyên nhân bước đầu được xác định là do rễ cây phượng bị mục rỗng. Sự việc này một lần nữa gióng lên hồi chuông cảnh báo về tình trạng cây đổ ngã do mục gốc. Trước đó, tại TP HCM, một cây phượng bật gốc trong sân trường đã làm 1 học sinh ở quận 3 tử vong. Xã hội Giao thông Phượng Người đi đường Đè Phan Rang – Tháp Chàm Cấp cứu Đường Nguyễn Văn Trỗi Người bị thương Ninh Thuận Thanh Sơn Bị thương Ngã Phương Dọn dẹp Rễ cây Chung tay Địa bàn Ùn tắc giao thông Rỗng Hiện trường Trơ"
        ]

data = []
stopwords = []
with open(STOP_WORDS, 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])

for i in range(len(test)):
    data.append(SplitWord(test[i], stopwords).segmentation_remove_stop_word())

print(data)

count_vector = cv.transform(data)
tf_idf_vector = tfidf_transformer.transform(count_vector)

# nb_model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
y_pred = text_clf.predict(tf_idf_vector)


print(y_pred)


# print(classification_report(y_test, y_pred))

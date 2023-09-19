# https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/notebook
# import tokenization
from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_text as text
import yaml



class bert_model():
    def __init__(self):
        self.model_multi_class = None
        self.model_multi_label = None
        self.mix_model = False

    def _load_model(self, model_multi_class_path = "", model_model_multi_label = ""):
        self.model_multi_class = load_model(model_multi_class_path, custom_objects={'KerasLayer':hub.KerasLayer})
        self.model_multi_label = load_model(model_model_multi_label, custom_objects={'KerasLayer':hub.KerasLayer})

    def decode_predict(self, predict_multi_class = None, predict_multi_label = None):
        probability_mclass = []
        probability_mlabel = []

        # first
        # [[0.03867007 0.01986488 0.04269431 0.01849338 0.49107984 0.01372945
        # 0.01034406 0.07810728 0.01236457 0.00373397 0.00449229 0.03290792
        # 0.18332264 0.03418403 0.00640096 0.00961033]]
        # second
        # [[0.02936517 0.04037957 0.14405015 0.03407615 0.27514702 0.11008038
        # 0.02703867 0.04628871 0.10379598 0.02202474 0.01588168 0.03402474
        # 0.05120867 0.21746662 0.09315041 0.04211371]]

        if (predict_multi_class is not None):

            keyword_value = []

            keyword_index = {}
            try:
                with open('./data/keyword/20230620-Test1-class16.yml', 'r', encoding="utf-8") as f:
                    keyword_index = yaml.safe_load(f)
            except Exception as e:
                print(e)
                ...

            if (self.mix_model is True):
                i = 0
                for p in predict_multi_class[1][0]:
                    _tmp = []
                    if (p >= 0.6):
                        # for key, value in keyword_index.items():
                            # if (value == i):
                            #   _tmp.append(key)  
                        keyword_value.append(1)
                    else:
                        keyword_value.append(0)
                    i += 1

                    # if (_tmp != []):
                    #     keyword_value.append([i, _tmp])

                predict_multi_class = predict_multi_class[0]

            max_index = np.argmax(predict_multi_class, axis=1)[0]

            max_probability = predict_multi_class[0][max_index]

            probability_mclass.append([self.transLabel(max_index), max_probability, keyword_value])
        else:
            probability_mclass.append(["None", 'None'])


        if (predict_multi_label is not None):
            result = predict_multi_label[0]

            i = 0
            for r in result:
                if (result[i] >= 0.0):
                    probability_mlabel.append([self.transLabel(i), result[i]])
                i += 1

            if (probability_mlabel == []):
                probability_mlabel.append(["None", 'None'])
        else:
            probability_mlabel.append(["None", 'None'])

        
        return probability_mclass, probability_mlabel

    def transLabel(self, number):
        label = {
            '0': '電信發票',
            '1': '國家漫遊方案',
            '2': '合約到期日',
            '3': '帳單折抵卷',
            '4': '本期帳單',
            '5': '通話收費明細',
            '6': '快速繳費',
            '7': '門號復話',
            '8': '小額收費代付',
            '9': 'Googleplay代收',
            '10': 'iTunesStore代收',
            '11': 'FriDay影音異動',
            '12': '單門號續約',
            '13': '新啟用/攜碼',
            '14': '線上續約優惠',
            '15': '家用寬頻'
        }

        return label.get(str(number))


input_data = [
    ['我想查看看我的發票有沒有中獎', 0],
    ['想知道有什麼漫遊方案', 1],
    ['想知道有什麼國際漫遊方案', 1],
    ['我想知道中國漫遊方案', 1],
    ['起問有中國漫遊方案嗎', 1],
    ['請問我可以知道合約到期日嗎?', 2],
    ['合約資訊及優惠', 2],
    ['請給我帳單折抵說明', 3],
    ['請給我帳單折抵卷說明', 3 ],
    ['我想問一下如何使用帳單折抵卷？', 3],
    ['我希望了解帳單折抵卷的使用期限', 3],
    ['可以告訴我如何獲得帳單折抵卷嗎', 3],
    ['我想了解帳單折扣卷的使用規則。', 3],
    ['我想要查詢帳單折抵說明', 3],
    ['想知道合約到期日', 2],
    ['想知道合約到期日', 2],
    ['想查本期帳單', 4],
    ['我只是來打雜的', 0],
    ['我只是隨便講句話', 0],
    ['信用卡嚕', 0],
    ['GooglePlay!!', 9],
    ['GooglePlay代收不能成功繳費', 9],
    ['隨便聊天唷唷唷', 0],
    ['我只是', 0],
    ['只是', 0],
    ['隨便講句話', 0],
    ['今天天氣如何', 0],
    ['', 2]
]

if __name__ == "__main__":
    model = bert_model()
    # 目前最好得版本，epoch20明顯比epoch5收斂
    # model._load_model('./save_model/20230628-V2-class16-multiclass_epoch20_bs4_myself.h5', './save_model/20230628-V2-class16-sigmoid_epoch20_bs4_myself.h5')
    model._load_model('./save_model/20230706-V3-class16-mix_model-binaryFocal_gamma4_alpha075_connectlabel_epoch20_bs4_myself.h5', './save_model/20230628-V2-class16-sigmoid_epoch20_bs4_myself.h5')
    # model._load_model('./save_model/20230628-V2-class16-multiclass_epoch20_bs4_myself.h5', './save_model/20230628-V2-class16-sigmoid_epoch20_bs4_myself.h5')

    model.mix_model = True

    userInput = input("請輸入想詢問的內容: ")
    
    while userInput != 'quit':
        y_predict_multi_class = model.model_multi_class.predict([userInput.lower()])
        y_predict_multi_label = model.model_multi_label.predict([userInput.lower()])

        # print('first')
        # print(y_predict_multi_class)

        # print('second')
        # print(y_predict_multi_label)
        # print()
        # sys.exit(0)

        probability_mclass, probability_mlabel = model.decode_predict(y_predict_multi_class, y_predict_multi_label)
        
        print('first')
        print(probability_mclass)

        print('second')
        print(probability_mlabel)
        print()
        userInput = input("請輸入想詢問的內容: ")
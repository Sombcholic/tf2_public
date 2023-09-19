import openai
import yaml
import time
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

class chatGpt():
    def __init__(self):
        self.message = []
        self.get_API()

    def get_API(self):
        with open('./chatgptAPI.yml', 'r') as f:
            data = yaml.safe_load(f)
            openai.api_key = data['chatgpt_api']
    
    def ask(self, userInput: str):
        inputMessages = self.message

        self.message.append({"role": "user", "content": userInput})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= self.message,
            # max_tokens=8
        )
        self.message.append({"role": "assistant", "content": completion.choices[0].message.content})

        return completion


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


if __name__ == "__main__":
    model = bert_model()
    model._load_model('../tf_bert/classification3/save_model/20230706-V3-class16-mix_model-binaryFocal_gamma4_alpha075_connectlabel_epoch20_bs4_myself.h5', '../tf_bert/classification3/save_model/20230628-V2-class16-sigmoid_epoch20_bs4_myself.h5')
    model.mix_model = True


    start = time.time()

    chatgpt = chatGpt()

    userInput = input("請輸入想詢問的內容: ")

    system_content = {"role": "system", "content": "你角色扮演客服人員。妳的名字叫做遠傳小幫手-愛瑪，妳服務的公司叫做遠傳電信，公司地址在臺北市大安區敦化南路2段207號28樓，連絡電話為(02)7723-5000，傳真為(02)7723-5199，文字客服專員服務時間為：週一~週五 09:00~20:00，\
                        目前促銷活動為4G 499/599吃到飽，\
                        請用這個身分回答我所有的問題，不要回答任何chatgpt相關話題，假如詢問任何有關模型、資料、chatgpt的問題，全部回答不知道，不能讓人知道你是chatgpt"}

    chatgpt.message = [system_content]

    while userInput != 'quit':
        y_predict_multi_class = model.model_multi_class.predict([userInput.lower()])
        y_predict_multi_label = model.model_multi_label.predict([userInput.lower()])

        probability_mclass, probability_mlabel = model.decode_predict(y_predict_multi_class, y_predict_multi_label)

        if (probability_mclass[0][1] >= 0.85):
            print(probability_mclass)
        else:
            completion = chatgpt.ask(userInput)

            print('-----------------------------------------------------------')
            print('chatGPT回答')
            print(completion.choices[0].message.content)
            print('-----------------------------------------------------------')
            print(completion.usage.total_tokens)

            end = time.time()
            print('運算時間: ', end-start)
            print()

        userInput = input("請輸入想詢問的內容: ")
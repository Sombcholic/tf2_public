Verison: https://www.tensorflow.org/install/source_windows#gpu  
cuda: https://developer.nvidia.com/cuda-toolkit-archive  
cudnn: https://developer.nvidia.com/rdp/cudnn-archive  

此專案使用以下版本並且要設定系統變數  
cuda: V11.2  
cudnn: V8.1  
python: 3.8 => pipenv --python 3.8  
tensorflow: 2.10.0  
tensorflow_hub: 0.13.0    

安裝tf-models-official需要更新C++  
https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst  

# 安裝library
請使用pipenv安裝，指令為pipenv install

# chatbot  
此專案結合bert以前chatgpt，假如bert可以有效辨識domain的class，對話將導引到相對印的class回覆語句，假如無法有效辨識，則利用chatgpt回覆  
注意：  
1. model folder: 底下要放置bert專屬的model以及preprocess向量，分別為bert_zh_L-12_H-768_A-12_4及bert_zh_preprocess_3  
2. 要新建一個yaml檔作為chatgpt的api key，檔案：chatgptAPI.yml，content: chatgpt_api: sk-XXXXX, chatgpt_api2:sk-XXXXXXX  

# DistilBERT  
此專案為使用DistilBERT進行分類，main_hospital_multilabel.py使用比較特別的概念，有主類別和次類別，主類別為癌症種類，次類別為關鍵字詞，透過結合主類別和次類別的訓練結果調整模型，發現模型有更好的表現，其中次類別不是concatenate最後的output，而是concatenate最後output的前一層layer  
  
提醒:  
1. 醫療DATA為利用chatgpt透過少許的domain data進行類比生成的資料，概念可參考這https://github.com/Sombcholic/chatgpt_public.git  

# tf_bert  
此專案為基本的bert模型





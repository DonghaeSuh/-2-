# NLP_Tensorflow2

Tensorflow 2로 구현된 NLP 프로젝트 repositoy


## 책 이름

![image](https://github.com/DonghaeSuh/NLP_tensorflow2/assets/82081872/ba8ce0e3-3f0c-4036-a899-d026fefe7079)
<br>
[텐서플로 2와 머신러닝으로 시작하는 자연어 처리](https://wikibook.co.kr/nlp-tf2/)



## 목차
- 0_babi : babi dataset을 이용한 Question and Answer
  
- 0_test : 여러가지 간단한 이해를 위한 도구코드 테스트
- 1_NLP_intro : EDA(탐험적 데이터 분석)과 간단한 텍스트 유사도 모델 구현(DENSE 사용)
- 2_NLP_basic : Tokenizer와 one-hot encoding 
- 3_Tools : Library(pandas,numpy,matplotlib,re,beautifulsoup),Tokenizer,sklearn,tensorflow_keras
- 4_Text_Classification : 텍스트 분류
- 5_Text_Similarity: 텍스트 유사도
- 6_ChatBot : 챗봇 (바나나우 어텐션(Bahdanau attention), 트랜스포머(transformer))
- 7_PRETRAIN_METHOD : Huggingface를 이용한 BERT와 GPT2 fine-tuning

## 저장소
```
NLP_tensorflow2
├── 0_Babi
│   ├── tasks_1-20_v1-2 
│   ├── Babi.ipynb 
│   ├── model.h5 
├── 0_Test 
│   ├── Pandas_tool : pandas example
│   │   ├── read_csv_test.ipynb
│   │   ├── test.csv
│   │   ├── test2.csv
│   │   └── test3.csv
│   ├── keras.layers_test : keras.layers tools
│   │   ├── append()&functools_reduce().ipynb
│   │   ├── layer_dot().ipynb
│   │   ├── layers_GlobalMaxPooling1D()_tf_concat()_ex.ipynb
│   │   └── layers_add()&layers_Activation().ipynb
│   ├── numpy_tool : numpy example
│   │   ├── np_stack().ipynb
│   │   ├── np_sum().ipynb
│   │   ├── numpy_matrix_operation.ipynb
│   │   ├── tf_reduce_sum().ipynb
│   │   └── tf_transpose_perm.ipynb
│   └── sklearn_test
│       └── TfidfVectorizer_test.ipynb
├── 1_NLP_intro
│   ├── EDA.ipynb : imdb dataset
│   └── Text_similarity.ipynb
├── 2_NLP_basic
│   ├── One_hot.ipynb
│   └── Tokenizer.ipynb
├── 3_Tools
│   ├── Library
│   │   ├── data_in
│   │   ├── Beautiful Soup.ipynb
│   │   ├── Matplotlib.ipynb
│   │   ├── Numpy.ipynb
│   │   ├── Pandas.ipynb
│   │   └── re.ipynb
│   ├── Tokenizer
│   │   ├── KoNLPy.ipynb
│   │   ├── Spacy.ipynb
│   │   └── nltk.ipynb
│   ├── scikit_learn
│   │   ├── feature_extraction.ipynb
│   │   ├── iris.ipynb
│   │   └── k-nn_and_k-means.ipynb
│   └── tensorflow_keras
│   │   ├── Sentiment Analysis.ipynb
│   │   ├── model_fit.ipynb
│   │   ├── tensorflow_2.0.ipynb
│   │   ├── tf.keras.layers.Conv1D.ipynb
│   │   ├── tf.keras.layers.Dense.ipynb
│   │   ├── tf.keras.layers.Dropout.ipynb
│   │   └── tf.keras.layers.MaxPool1D.ipynb
├── 4_Text_Classification
│   ├── data_in
│   ├── data_out
│   ├── 300features_40minwords_10context
│   ├── CNN.ipynb
│   ├── Kr_CNN.ipynb
│   ├── Kr_data.ipynb
│   ├── LSTM.ipynb
│   ├── Logistic_tfidf.ipynb
│   ├── Logistic_word2vec.ipynb
│   ├── RandomForest.ipynb
│   └── en_Text_Classification.ipynb
├── 5_Text_Similarit
│   ├── .ipynb
│   ├── CNN.ipynb
│   ├── EDA&Preprocessing.ipynb
│   ├── MaLSTM.ipynb
│   └── XG_boost.ipynb
├── 6_ChatBot
│   ├── EDA.ipynb
│   ├── Preprocess.ipynb
│   ├── Transformer_ChatBot.ipynb
│   ├── preprocess.py
│   └── seq2seq.ipynb
├── 7_PRETRAIN_METHOD
│   ├── GPT2
│   │   ├── GPT2_KorNLI.ipynb
│   │   ├── GPT2_KorSTS.ipynb
│   │   ├── GPT2_LM.ipynb
│   │   └── GPT2_NSMC.ipynb
│   ├── BERT_Classification_NSMC.ipynb
│   ├── KorNLI_EDA.ipynb
│   ├── KorNLI_fine_tuning.ipynb
│   ├── KorQuAD_EDA.ipynb
│   ├── KorQuAD_finetuning.ipynb
│   ├── KorSTS_EDA.ipynb
│   ├── KorSTS_finetuning.ipynb
│   ├── NER_EDA.ipynb
│   └── NER_finetuning.ipynb
├── .gitignore
└── readme.md
```

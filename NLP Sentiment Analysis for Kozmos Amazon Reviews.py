
# Amazon Yorumları için Duygu Analizi

# İş Problemi

# Amazon üzerinden satışlarını gerçekleştiren ev tekstili ve günlük giyim odaklı üretimler yapan Kozmos,
# ürünlerine gelen yorumları analiz ederek ve aldığı şikayetlere göre özelliklerini geliştirerek satışlarını
# artırmayı hedeflemektedir. Bu hedef doğrultusunda yorumlara duygu analizi yapılarak etiketlenecek ve
# etiketlenen veri ile sınıflandırma modeli oluşturulacaktır.


# Veri Seti Hikayesi

# Veri seti belirli bir ürün grubuna ait yapılan yorumları, yorum başlığını, yıldız sayısını ve yapılan yorumu
# kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

# 4 Değişken 5611 Gözlem 489 KB
# Star      : Ürüne verilen yıldız sayısı
# HelpFul   : Yorumu faydalı bulan kişi sayısı
# Title     : Yorum içeriğine verilen başlık, kısa yorum
# Review    : Ürüne yapılan yorum


# Proje Görevleri

####################################
# Görev 1: Metin Ön İşleme
#####################################
############################################
# Adım 1: amazon.xlsx verisini okutunuz.
############################################
# !pip install nltk
# !pip install textblob
# !pip install wordcloud

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_excel(r"C:\Users\GunalHincal\Desktop\NLP\Case1 NLP Proje\amazon.xlsx")
df.head()
#    Star  HelpFul                                          Title                                             Review
# 0     5        0                                    looks great                                      Happy with it
# 1     5        0  Pattern did not align between the two panels.  Good quality material however the panels are m...
# 2     5        0               Imagery is stretched. Still fun.  Product was fun for bedroom windows.<br />Imag...
# 3     5        0                 Que se ven elegantes muy finas   Lo unico que me gustaria es que sean un poco ...
# 4     5        0                             Wow great purchase  Great bang for the buck I can't believe the qu...
df.info()

df.columns
# Index(['Star', 'HelpFul', 'Title', 'Review'], dtype='object')
##############################################################
# Adım 2: Review değişkeni üzerinde ;
# a. Tüm harfleri küçük harfe çeviriniz.
# b. Noktalama işaretlerini çıkarınız.
# c. Yorumlarda bulunan sayısal ifadeleri çıkarınız.
# d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız.
# e. 1000'den az geçen kelimeleri veriden çıkarınız.
# f. Lemmatization işlemini uygulayınız.
##############################################################


# Normalizing Case Folding
###############################
df['Review']
df['Review'] = df['Review'].str.lower()


# Punctuations
###############################
df['Review'] = df['Review'].str.replace('[^\w\s]', '')


# Numbers
###############################
df['Review'] = df['Review'].str.replace('\d', '')


# Stopwords
###############################
import nltk
# nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# Rarewords
###############################

# drops adında bir liste oluşturup 1000 den az geçen kelimleri bu listede tutuyoruz
drops = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
drops

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
df['Review'].value_counts()


# Lemmatization
###############################
# nltk.download('wordnet')
import nltk
nltk.download('omw-1.4')

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


#####################################
# Görev 2: Metin Görselleştirme
######################################
#####################################################################################################
# Adım 1: Barplot görselleştirme işlemi için;
# a. "Review" değişkeninin içerdiği kelimelerin frekanslarını hesaplayınız, tf olarak kaydediniz.
# b. tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
# c. "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini
# tamamlayınız.
#############################################################################################################

# Terim Frekanslarının Hesaplanması
###############################

tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)


# Barplot
###############################
tf[tf["tf"] > 500]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

####################################################################################
# Adım 2: WordCloud görselleştirme işlemi için;
# a. "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz.
# b. WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz.
# c. Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
# d. Görselleştirme adımlarını tamamlayınız. (figure, imshow, axis, show)
#########################################################################################

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      min_font_size=5,
                      max_words=500,
                      background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# wordcloud.to_file dediğimizde word cloud png olarak bunu çalışma dizini ortamınıza kaydedebilirsiniz.
wordcloud.to_file("wordcloud_kozmos_words.png")


# bir de bu wordcloud u şablonla yapalım
import numpy as np
amazon_mask = np.array(Image.open(r"C:\Users\GunalHincal\Desktop\NLP\Case1 NLP Proje\amazon.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=amazon_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

wc.to_file("wc_amazon_template.png")

####################################
# Görev 3: Duygu Analizi
#######################################
########################################
# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz.
###################################################################################################

sia = SentimentIntensityAnalyzer()


#############################################################################################
# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarını inceleyiniz;
# a. "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız.
# b. İncelenen ilk 10 gözlem için compound skorlarına göre filtreleyerek tekrar gözlemleyiniz.
# c. 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz.
# d. "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e
# ekleyiniz.

# NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorumsınıflandırma makine öğrenmesi modeli için
# bağımlı değişken oluşturulmuş oldu.
#######################################################################################################

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# review i labellıyoruz bu bölümde eğer compound 0 dan büyükse pos yaz 0 dan küçükse neg olarak değerlendir
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# yeni bir değişken oluşturarak bu etiketimizi sayısal ifade ediliş formuna getiriyoruz pos için 1 neg için 0 yazıyoruz
df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()
# pos    4764
# neg     847
# Name: sentiment_label, dtype: int64

df.groupby("sentiment_label")["Star"].mean()
# sentiment_label
# neg   3.39
# pos   4.59
# Name: Star, dtype: float64


#######################################################
# Görev 4: Makine Öğrenmesine Hazırlık
##########################################################
##########################################################
# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.
####################################################################################

# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"],
                                                    random_state=42)


"""
# ["sentiment_label"] değişkenini label encoder a sokarak tek değişken haline getiriyoruz
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])
y = df["sentiment_label"]  # bağımlı değişken
X = df["reviewText"]  # bağımsız değişken
"""

#################################################################################################
# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
# a. TfidfVectorizer kullanarak bir nesne oluşturunuz.
# b. Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
# c. Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.
####################################################################################################

# kendi veri setimize tf idf i uygulayalım
from sklearn.feature_extraction.text import TfidfVectorizer

# tf idf vektorizer ımızı "tf_idf_word_vectorizer" adıyla oluşturuyoruz train setine fit ediyoruz
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)

# x train setine de transform ediyoruz
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)

# x test setine transform ediyoruz
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


############################################
# Görev 5: Modelleme (Lojistik Regresyon)
############################################
###########################################
# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
################################################################################

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)


#####################################################################################3
# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
# a. Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
# b. classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
# c. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
############################################################################################

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

#            precision    recall  f1-score   support
#          neg       0.34      0.92      0.50        83
#          pos       0.99      0.89      0.94      1320
#     accuracy                           0.89      1403
#    macro avg       0.67      0.90      0.72      1403
# weighted avg       0.96      0.89      0.91      1403

cross_val_score(log_model,
                x_train_tf_idf_word,
                train_y,
                scoring="accuracy",
                cv=5).mean()

# 0.8835562233626408

"""
cross_val_score(log_model,
                x_test_tf_idf_word,
                test_y,
                scoring="accuracy",
                cv=5).mean()
# 0.8553152008134214
"""
#####################################################################################
# Adım 3: Veride bulunan yorumlardan rastgele seçerek modele sorulması;
# a. sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayınız.
# b. Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
# c. Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
# d. Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
# e. Örneklemi ve tahmin sonucunu ekrana yazdırınız.
##########################################################################################

random_review = pd.Series(df["Review"].sample(1).values)
random_review  # 0    love

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# bu yorumu modelimize sormamız lazım
new_review = CountVectorizer().fit(train_x).transform(random_review)

pred = log_model.predict(new_review)
pred  # array(['pos']  yorum olumluymuş

print(f'Review:  {random_review[0]} \n Prediction: {pred}')
# Review:  love
#  Prediction: ['pos']

####################################################
# Görev 6: Modelleme (Random Forest)
######################################################
#######################################################
# Adım 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
# a. RandomForestClassifier modelini kurup fit ediniz.
# b. Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
# c. Lojistik regresyon modeli ile sonuçları karşılaştırınız.
#####################################################################


# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model,
                x_train_tf_idf_word,
                train_y,
                cv=5,
                n_jobs=-1).mean()

# 0.9118307297330123


# SONUÇLAR

# Lojistik regresyon modeli    # 0.8835562233626408
# Random Forest modeli         # 0.9118307297330123






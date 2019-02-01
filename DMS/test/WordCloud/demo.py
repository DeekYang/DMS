#-*- coding: utf-8 -*-
#中文分词+词云显示
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jieba
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS

mytext = open("doc.txt").read()
# http://jirkavinse.deviantart.com/art/quot-Real-Life-quot-Alice-282261010
pic = np.array(Image.open("pic.png"))

# jieba 分词
mytext = " ".join(jieba.cut(mytext))

stopwords = set(STOPWORDS)
stopwords.add("禁用词")

# 通过 mask 参数来设置词云形状
wordcloud = WordCloud(font_path="simsun.ttf",mask=pic,stopwords=stopwords,
	background_color="white",max_words=2000,max_font_size=40,random_state=42).generate(mytext)

image_colors = ImageColorGenerator(pic)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure()

# 使用图片颜色布局
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")
plt.show()
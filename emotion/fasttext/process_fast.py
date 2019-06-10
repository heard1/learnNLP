# encoding=utf-8

"""
import fasttext
model=fasttext.skipgram('train.csv','ski')
# model = fasttext.load_model('ski.bin')
print(model['你好'])

c = fasttext.supervised('train.csv','model',label_prefix='label')
re=c.test('valid.csv')
print(re.precision) #86%
"""
from flask import Flask, request
import fasttext
import re
import jieba
import logging
import time


classifier = fasttext.load_model('model.bin', label_prefix='label')
regex = re.compile(u'[^\u4E00-\u9FA5]')
words = "分词启动中。。。"
words = ' '.join(jieba.cut(regex.sub('', words)))
print("initialization success!")

def create_app():
    app = Flask(__name__)
    @app.route('/', methods=['GET', 'POST'])
    def callback():
        start_time = time.time()
        s = request.args.get("s") or "EOF"
        app.logger.warning('the sentence is %s', s)
        try:
            ans = classifier.predict([' '.join(jieba.cut(regex.sub('', s)))])
            app.logger.warning('the answer is %s', ans[0][0])
            end_time = time.time()
            cur_time = " %.2fms"%(1000*(end_time-start_time))
            return ans[0][0]+cur_time
        except:
            end_time = time.time()
            cur_time = " %.2fms"%(1000*(end_time-start_time))
            app.logger.warning('the answer is %s', 'error')
            return "请输入中文语句！"+cur_time
    return app

app = create_app()

if __name__ == '__main__':
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0',port=19610)
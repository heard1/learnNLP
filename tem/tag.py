# encoding=utf-8
from flask import Flask, request
import logging
import jieba.posseg as pseg

def get_postag(text):
    word = []
    postag = []
    res = list(pseg.cut(text))
    for item in res:
        word.append(item.word)
        postag.append(item.flag)
    postag_dict = {
        'word':word,
        'postag':postag
    }
    return postag_dict

def create_app():
    app = Flask(__name__)
    @app.route('/', methods=['GET', 'POST'])
    def callback():
        s = request.args.get("s") or "EOF"
        app.logger.warning('the sentence is %s', s)
        ans = get_postag(s)
        app.logger.warning('the answer is %s', ans)
        return str(ans)
    return app

app = create_app()

if __name__ == '__main__':
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0',port=8888)
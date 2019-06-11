import json

with open('comment.json', 'r', encoding='utf-8') as f:
    dict = json.load(fp=f)

comment, star = [], []
for single in dict:
    comment.append(single['comment'])
    star.append(single['star'])
with open('com.json', 'w') as f:
    json.dump(comment, f)
with open('star.json', 'w') as f:
    json.dump(star, f)

with open('star.json', 'r', encoding='utf-8') as f:
    star = json.load(fp=f)

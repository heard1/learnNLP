
tot = []
tag_set = set()
with open('store.txt', encoding='utf-8') as f:
    for line in f.readlines():
        tot.append(eval(line))

number = []
with open('qa.txt', encoding='utf-8') as f:
    for line in f.readlines():
        num,s = line.split('\t')
        number.append(num)

for single in tot:
    for key in single:
        tag_set.add(key)

tag = {}
for num,i in enumerate(tag_set):
    tag[i] = num

f = open('SQL_file.txt','w', encoding='utf-8')
for num, question in enumerate(tot):
    topic_id = number[num]
    for key in question:
        tag_id = tag[key]
        score = question[key]
        f.write(str(tag_id)+'\t'+str(topic_id)+'\t'+str(score)+'\n')
f.close()

f = open('SQL_file2.txt','w', encoding='utf-8')
for key in tag:
    f.write(str(tag[key]) + '\t' + str(key) + '\n')
f.close()
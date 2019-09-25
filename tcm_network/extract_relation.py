disease = set()
plant = set()
symptom = set()
drug = set()
with open('disease.txt', encoding='utf-8') as f:
    for line in f.readlines():
        disease.add(line[:-1])
with open('plant.txt', encoding='utf-8') as f:
    for line in f.readlines():
        plant.add(line[:-1])
with open('symptom.txt', encoding='utf-8') as f:
    for line in f.readlines():
        symptom.add(line[:-1])
with open('drug.txt', encoding='utf-8') as f:
    for line in f.readlines():
        drug.add(line[:-1])

def extract(sentence):
    cur_disease = []
    cur_plant = []
    cur_symptom = []
    cur_drug = []
    for i in disease:
        if i in sentence:
            cur_disease.append(i)
    for i in plant:
        if i in sentence:
            cur_plant.append(i)
    for i in symptom:
        if i in sentence:
            cur_symptom.append(i)
    for i in drug:
        if i in sentence:
            cur_drug.append(i)
    for i in cur_symptom:
        if i in cur_disease:
            cur_symptom.remove(i)


    for i in cur_disease:
        for j in cur_plant:
            print(j, i, '治疗')
    for i in cur_disease:
        for j in cur_drug:
            print(j, i, '治疗')
    for i in cur_plant:
        for j in cur_symptom:
            print(j, i, '治疗')
    for i in cur_symptom:
        for j in cur_drug:
            print(j, i, '治疗')
    for i in cur_disease:
        for j in cur_symptom:
            print(j, i, '症状')
    for i in cur_plant:
        for j in cur_drug:
            print(j, i, '组成')

    for i in range(len(cur_symptom)):
        for j in cur_symptom[i+1:]:
            print(cur_symptom[i], j, '伴随')
    for i in range(len(cur_drug)):
        for j in cur_drug[i+1:]:
            print(cur_drug[i], j, '伴随')
    for i in range(len(cur_plant)):
        for j in cur_plant[i+1:]:
            print(cur_plant[i], j, '伴随')
    for i in range(len(cur_disease)):
        for j in cur_disease[i+1:]:
            print(cur_disease[i], j, '伴随')
    print("===============================")



if __name__ == "__main__":
    while True:
        """
        example:
        我肚子疼，耳痛，还有点近视，能不能吃建兰做成的曼宁治疗？
        难产造成的龋齿能吃去痛片吗？
        """
        extract(input('>>>'))
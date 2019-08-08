import os
from elasticsearch import Elasticsearch


class CrimeQA:
    def __init__(self):
        self._index = "qa_data"
        self.es = Elasticsearch([{"host": "192.168.110.8", "port": 9201}])
        self.doc_type = "qa"
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.embedding_path = os.path.join(cur, 'embedding/word_vec_300.bin')

        self.min_score = 0.4
        self.min_sim = 0.8

    '''根据question进行事件的匹配查询'''

    def search_specific(self, value, key="question"):
        query_body = {
            "query": {
                "match": {
                    key: value,
                }
            }
        }
        searched = self.es.search(index=self._index, doc_type=self.doc_type, body=query_body, size=20)
        # 输出查询到的结果

        return searched["hits"]["hits"]

    '''基于ES的问题查询'''

    def search_es(self, question):
        answers = []
        res = self.search_specific(question)
        for hit in res:
            answer_dict = {}
            answer_dict['score'] = hit['_score']
            answer_dict['sim_question'] = hit['_source']['question']
            answer_dict['answer'] = hit['_source']['answer']
            answers.append(answer_dict)

        return answers

    '''问答主函数'''

    def search_main(self, question):
        candi_answers = self.search_es(question)


        try:
            return candi_answers[0]['answer']
        except:
            return "您好，对于此类问题，您可以直接准备后事了"

if __name__ == "__main__":
    handler = CrimeQA()
    while (1):
        question = input('question:')
        final_answer = handler.search_main(question)
        print('answers:', final_answer)


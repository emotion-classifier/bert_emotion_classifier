import naive_bayes_api
import support_vector_api_update
import biLSTM_api
import bert_api
from konlpy.tag import Okt

def get_morphs(text):
    tokenizer = Okt()
    morphs = tokenizer.morphs(text)
    return [n for n in morphs]

while True:
    print("트윗을 입력해주세요 : ", end="")
    input_text = input()
    naive_bayes = naive_bayes_api.call_api(input_text)
    svm = support_vector_api_update.call_api(input_text)
    bilstm = biLSTM_api.call_api(input_text)
    bert = bert_api.call_api(input_text)
    print(input_text)
    print("navie_bayes predict : \t\t\t\t", naive_bayes)
    print("support vector machine predict : \t", svm)
    print("biLSTM predict : \t\t\t\t\t", bilstm)
    print("bert predict : \t\t\t\t\t\t", bert)
    print("\n")

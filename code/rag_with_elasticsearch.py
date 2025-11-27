import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

es_password = os.getenv("es_password")
openai_key = os.getenv("openai_key")

print(f"es_password: {es_password}")
print(f"openai_key: {openai_key}")

index_name = "test"

# --- SentenceTransformer를 이용한 임베딩 생성 ---

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)

# ------------------------------------------------------

# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index=index_name, query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index=index_name, knn=knn)

es_username = "elastic"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer", # nori_tokenizer 를 사용, 몇가지 설정값이 필요..
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                # "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                "stoptags": ["E", "SC", "SE", "SF"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine", #"l2_norm"
        }
    }
}
# =============================== 여기부터 벡터 임베딩 생성 및 색인

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index(index_name, settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("../data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    # embedding이 list인지 확인
    if isinstance(embedding, list):
        doc["embeddings"] = embedding
    else:
        doc["embeddings"] = embedding.tolist()  # numpy 배열일 경우
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add(index_name, index_docs)
print(ret)

# =============================== 여기까지가 벡터 임베딩 생성 및 색인

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

print(f">>>> query : {test_query}")
# 역색인을 사용하는 검색 예제
search_result_retrieve = sparse_retrieve(test_query, 3)

print(f">>>> spase")
# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제
search_result_retrieve = dense_retrieve(test_query, 3)

print(f">>>> dense")
# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

print(f">>>>"*6)
# =============================== 여기까지가 생성된 색인 기반 쿼리 테스트

# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback
import httpx

# =============================== OpenAI

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = openai_key

client = OpenAI(http_client=httpx.Client())
# client = OpenAI()

# 사용할 모델을 설정
llm_model = "gpt-4o-mini"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 상식 전문가

## Instruction
- 모든 대화는 반드시 standalone_query를 생성하고, search 함수를 호출하는 JSON 형식으로 반환한다.
- 사용자 메시지 내 '우울해', '기분이 좋아' 등 기분에 대한 내용만 있다면 일상 대화이며, 일상 대화는 standalone_query를 생성하지 않는다.
- 사용자 메시지 내 '?', '연구', '코드', '알고리즘', '개발', '네트워크', '실험', '사회' 등의 단어가 포함되어 있다면 search 도구를 우선 호출한다.
- 사용자 메시지 내 자연현상, 물리, 화학, 생물, 식물, 영약학, 의학, 인간 생리학, 인간 발달 및 노화(human aging), 인간 성(human sexuality), IT/컴퓨터 과학, 역사, 사건, 코드 지식에 관한 주제가 포함된다면 search 도구를 우선 호출한다.
- 사용자 메시지 내 영어 단어가 포함된 경우, 한국어로 standalone_query를 생성한다.
- 사용자 메시지 내 참조되는 단어('그거', '저것')는 이전 문맥을 참고하여 구체적인 질문으로 변환한다.
- 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]

def normalize_recursive(obj):
    if isinstance(obj, str):
        # 문자열이면 직접 정규화
        obj = obj.replace("네트웍", "네트워크")
        obj = obj.replace("Dmitri Ivanovsky", "드미트리 이바노프스키")
        obj = obj.replace("python", "파이썬")
        obj = obj.replace("Python", "파이썬")
        return obj
    elif isinstance(obj, list):
        # 리스트면 각 요소에 재귀 적용
        return [normalize_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        # 딕셔너리면 값(value)에 재귀 적용
        return {key: normalize_recursive(value) for key, value in obj.items()}
    else:
        # 그 외 타입은 그대로 반환
        return obj
    
# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=60 #10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성 # function call 호출이 필요하다고 판단
    print(f"tool_calls - {result.choices[0].message.tool_calls}")
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        search_result = sparse_retrieve(standalone_query, 3)
        # search_result = dense_retrieve(standalone_query, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=120 #30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)

            # 테스트 검증하고 싶은 id만 체크해서 실행
            check_ids = []

            ## multi-turn
            # check_ids = [107, 42, 43, 97, 243, 66, 98, 295, 290, 68, 86, 89, 306, 39, 33, 249, 54, 3, 44, 278]

            ## chitchat
            # check_ids = [276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218]
            
            current_id = j["eval_id"]
            question = j["msg"]

            question = normalize_recursive(question)

            if check_ids:
                if current_id in check_ids:
                    print(f'Test {idx}\neval: {current_id}\nQuestion: {question}')
                    response = answer_question(question)
                    if "standalone_query" in response:
                        print(f'standalone_query: {response["standalone_query"]}')
                    print(f'Answer: {response["answer"]}\n')

                    # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
                    output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
                    of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
                    print(f"======================================")
            else:
                print(f'Test {idx}\neval: {current_id}\nQuestion: {question}')
                response = answer_question(question)
                if "standalone_query" in response:
                        print(f'standalone_query: {response["standalone_query"]}')
                print(f'Answer: {response["answer"]}\n')

                # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
                output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
                of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
                print(f"======================================")
            
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("../data/eval.jsonl", "sample_submission.csv")


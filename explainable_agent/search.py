import requests
import json


def main():
    url = "https://qianfan.baidubce.com/v2/ai_search/web_search"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "今天热点新闻"
            }
        ],
        "edition": "standard",
        "search_source": "baidu_search_v2",
        "search_recency_filter": "week"
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-LABxxxxxxxxxxxxxxx' #已脱敏，使用新的API密钥
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    response.encoding = "utf-8"
    print(response.text)


if __name__ == '__main__':
    main()

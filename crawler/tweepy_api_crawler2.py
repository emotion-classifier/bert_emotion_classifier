# -*- coding: utf-8 -*-

import tweepy
import os

consumer_key = "yeEBDjQEOZ7Qrg6oIPCTMWaV4"
consumer_secret = "ygYHoKbaqvvkMcE68CgNuSf40sgh1aNiqZY6Tti244Nu7387sz"
access_token = "848417898846576640-JXLS2Zw6Mce66O7WIwCqilm5Mfazvvr"
access_token_secret = "SVzXEqEFGF8k2ut8nrGca6az0KaHcObTN6buILzlcY5WO"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

location = "%s,%s,%s" % ("35.81905", "125.6320891", "500km")  # 검색기준(대한민국 중심) 좌표, 반지름
keyword = "짜증 OR 행복"                                      # OR 로 검색어 묶어줌, 검색어 5개(반드시 OR 대문자로)
wfile = open(os.getcwd()+"/twitter.txt", mode='w')        # 텍스트 파일로 출력(쓰기모드)

# twitter 검색 cursor 선언
cursor = tweepy.Cursor(api.search,
                       q=keyword,
                       since='2015-01-01', # 2015-01-01 이후에 작성된 트윗들로 가져옴
                       count=100,  # 페이지당 반환할 트위터 수 최대 100
                       geocode=location,
                       include_entities=True)
data = dict()
try:
    for i, tweet in enumerate(cursor.items()):
        if ("ATEEZ" in tweet.text or
                "방탄" in tweet.text or
                "강다니엘" in tweet.text or
                "냉장고를부탁해" in tweet.text or
                "아이잭" in tweet.text):
            continue
        print("{}: {}".format(i, tweet.text))
        if tweet.text not in data.keys():
            data[tweet.text] = ''
except:
    pass
for x in data.keys():
    wfile.write(x + '\n\n$$$$$$$$$$$$$$$$$$$$\n\n')
wfile.close()

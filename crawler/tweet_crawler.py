from selenium import webdriver
import time
import random
import datetime as dt
import re
import csv

browser = webdriver.Firefox(executable_path='C:\\geckodriver.exe')

f = open("tweetlist(park).csv", 'w', encoding='utf-8', newline='')
f2 = open("sung", 'r', encoding='utf-8')
namelist = f2.read().split("\n")
random.shuffle(namelist)
wr = csv.writer(f)
days = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
tweetlist = []
allcount = 0

for name in namelist:
    startdate = dt.date(year=2019, month=1, day=1)
    untildate = dt.date(year=2019, month=2, day=1)
    enddate = dt.date(year=2019, month=7, day=1)
    tweets = set()
    datecount = 0
    flag = False
    if allcount > 10000:
        break
    while enddate >= startdate:
        url = 'https://twitter.com/search?q=' + name + ' since%3A' + str(startdate) + '%20until%3A' + str(
            untildate) + '&src=typd'

        browser.get(url)
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        if datecount> 50:
            break
        for x in range(10):
            elements = browser.find_elements_by_class_name("TweetTextSize")
            for y in elements:
                if y.text not in tweets:
                    html_tag_removed_twits = re.sub('<.+?>', "", y.text, 0).strip()
                    url_removed_twits = re.sub('http.*', '', str(html_tag_removed_twits), 0).strip()
                    user_tag_removed_twits = re.sub('#.*', '', str(url_removed_twits), 0).strip()
                    user_circle_removed_twits = re.sub('@.*', '', str(user_tag_removed_twits), 0).strip()
                    pic_removed_twits = re.sub('pic.*', '', str(user_circle_removed_twits), 0).strip()

                    result = str(pic_removed_twits)
                    tweets.add(result)
                    datecount += 1
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            newHeight = browser.execute_script("return document.body.scrollHeight")

            if newHeight != lastHeight:
                lastHeight = newHeight
                continue
            else:
                startdate = untildate
                untildate += dt.timedelta(days=days[int(str(startdate)[5:7])])
                if datecount < 10:
                    flag = True
                    tweets.clear()
                    break
                elif datecount > 50:
                    break
        if flag:
            break
        lastHeight = newHeight
    if not flag:
        for x in tweets:
            wr.writerow([x])
            allcount += 1
        tweets.clear()
f.close()
f2.close()
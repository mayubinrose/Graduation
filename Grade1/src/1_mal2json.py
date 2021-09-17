import re
import os
from virus_total_apis import PublicApi as VirusTotalPublicApi
import json

all_samples_name = os.listdir('../data/VirusShare_00376')

# print(all_samples_name)
# 贪婪匹配以及避免转义操作
name_pattern = r'virusshare_(.+)'

API_KEY = 'ba5c6c0aac57377096a8f2bf53da561d7d9a21c80ddb92a0327fdc94b2a3e56b'


vt = VirusTotalPublicApi(API_KEY)


all_jsons = []
count = 0
num = 1
for sample_name in all_samples_name:
    md5 = re.findall(name_pattern, sample_name, re.IGNORECASE)
    if len(md5):
        print(md5)
        response = vt.get_file_report(md5)
        try:
            response = response['results']
        except:
            print("keyerror")
            continue
        engines = response['scans']
        av_lables = []
        for engine, result in engines.items():
            if result['result'] is None:
                continue
            list = []
            list.append(engine)
            list.append(result['result'])
            av_lables.append(list)
        sha1 = response['sha1']
        sha256 = response['sha256']
        md5 = response['md5']
        scan_date = response['scan_date']
        file_json = dict(sha1=sha1, av_labels=av_lables, scan_date=scan_date, sha256=sha256, md5=md5)
        all_jsons.append(file_json)
        count+=1
    if count % 100 == 0:
        json_str = json.dumps(all_jsons)
        file_path = '../json/'
        filename = file_path + 'allsample' + '.json'
        print("第{}轮开始写入".format(num))
        num+=1
        with open(filename, 'a') as fn:
            for one_json in all_jsons:
                json_str = json.dumps(one_json)
                fn.writelines(json_str)
                fn.write('\n')
        all_jsons.clear()
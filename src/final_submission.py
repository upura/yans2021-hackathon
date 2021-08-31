import os

import pandas as pd
from tqdm import tqdm

IS_LB = False


if __name__ == "__main__":
    city_df = pd.read_json("system_result/City.json", lines=True)
    if IS_LB:
        path = "./yans2021hackathon_tknzd_tohoku_bert/City/tokenized/leaderboard"
        files = os.listdir(path)
        files = [f[:-4] for f in files]
    else:
        files = pd.read_table("city_target.txt", header=None)[0].to_list()
    city_df = city_df[city_df.page_id.isin(files)]

    res = []
    for index, row in tqdm(city_df.iterrows(), total=len(city_df)):
        title = row["title"]
        page_id = row["page_id"]
        result = row["result"]
        for attribute in result:
            for ann in result[attribute]:
                text_offset = ann["text_offset"]
                html_offset = ann["html_offset"]
                system = ann["system"]
                data = {
                    "title": title,
                    "page_id": page_id,
                    "attribute": attribute,
                    "text_offset": text_offset,
                    "html_offset": html_offset,
                    "ENE": "1.5.1.1",
                }
                ths = 0
                if "01010" in system:
                    ths += 0
                if "02010" in system:
                    ths += 1
                if "03010" in system:
                    ths += 1
                if "05010" in system:
                    ths += 0
                if "07011" in system:
                    ths += 1
                if "10011" in system:
                    ths += 1
                if ths >= 2:
                    res.append(data)
    pd.DataFrame(res).to_json(
        "City.json", orient="records", force_ascii=False, lines=True
    )

    company_df = pd.read_json("system_result/Company.json", lines=True, chunksize=10000)
    if IS_LB:
        path = "./yans2021hackathon_tknzd_tohoku_bert/Company/tokenized/leaderboard"
        files = os.listdir(path)
        files = [f[:-4] for f in files]
    else:
        files = pd.read_table("company_target.txt", header=None)[0].to_list()

    res = []
    for df in company_df:
        df = df[df.page_id.isin(files)]
        for index, row in tqdm(df.iterrows(), total=len(df)):
            title = row["title"]
            page_id = row["page_id"]
            result = row["result"]
            for attribute in result:
                for ann in result[attribute]:
                    text_offset = ann["text_offset"]
                    html_offset = ann["html_offset"]
                    system = ann["system"]
                    if len(system) >= 3:
                        res.append(
                            {
                                "title": title,
                                "page_id": page_id,
                                "attribute": attribute,
                                "text_offset": text_offset,
                                "html_offset": html_offset,
                                "ENE": "1.4.6.2",
                            }
                        )
    pd.DataFrame(res).to_json(
        "Company.json", orient="records", force_ascii=False, lines=True
    )

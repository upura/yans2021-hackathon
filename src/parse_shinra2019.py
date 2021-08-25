import os

import pandas as pd


def convert_shinra2019_to_yans2021(category: str):
    assert category in ["City", "Company"]
    # escape oom
    dfs = pd.read_json("Company.json", lines=True, chunksize=10000)
    # get page_ids used in lb
    path = f"./yans2021hackathon_tknzd_tohoku_bert/{category}/tokenized/leaderboard"
    files = os.listdir(path)
    # remove an extention(.txt) from filenames
    files = [f[:-4] for f in files]
    res = []

    for df in dfs:
        # filter data in lb
        df = df[df.page_id.isin(files)]
        for index, row in df.iterrows():
            title = row["title"]
            page_id = row["page_id"]
            result = row["result"]
            for attribute in result:
                for ann in result[attribute]:
                    text_offset = ann["text_offset"]
                    html_offset = ann["html_offset"]
                    system = ann["system"]
                    if len(system) > 3:
                        res.append(
                            {
                                "title": title,
                                "page_id": page_id,
                                "attribute": attribute,
                                "text_offset": text_offset,
                                "html_offset": html_offset,
                                "ENE": "1.5.1.1" if category == "City" else "1.4.6.2",
                            }
                        )
        pd.DataFrame(res).to_json(
            f"lb_{category}.json", orient="records", force_ascii=False, lines=True
        )


if __name__ == "__main__":
    convert_shinra2019_to_yans2021("City")
    convert_shinra2019_to_yans2021("Company")

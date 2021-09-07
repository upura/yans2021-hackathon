# NLP若手の会 (YANS) 第16回シンポジウム ハッカソン Aチーム

- [NLP若手の会 (YANS) 第16回シンポジウム ハッカソン](https://yans.anlp.jp/entry/yans2021hackathon) におけるAチームのソースコードです。
- 運営により公開された[BERTベースライン](https://github.com/ujiuji1259/shinra-attribute-extraction)を改修しています。
- [リーダーボード](https://yans2021hackathon.pythonanywhere.com/)順位は1位、最終順位は2位でした。

## 取り組みの概要

[発表資料](https://drive.google.com/file/d/1p-WtGY2N3loPJhTWA_O9vSgZkxqifd-L/view?usp=sharing)をご覧ください。
公式の[開催報告](https://yans.anlp.jp/entry/yans2021report)では、全チームの発表資料が公開されています。

## 学習
`sh train.sh`

### train.shの例
```bash
python train.py \
    --input_path /path/to/Target_Category \
    --save_path /path/to/save_directory \
    --additional_name "" \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --grad_clip 1.0
```

## 予測
`sh predict.sh`.
前処理済みのデータ（１カテゴリ）を入力に，森羅2020の出力形式で予測結果を出力.
※ `model_path`はモデルファイルへのパスです．

### predict.shの例
```bash
python predict.py \
    --input_path /path/to/Target_Category \
    --model_path /path/to/model_file \
    --mode all
```

## Reference
- [NLP若手の会 (YANS) 第16回シンポジウム ハッカソン - NLP 若手の会](https://yans.anlp.jp/entry/yans2021hackathon)
- [森羅2020-JPでNER入門 - うしのおちちの備忘録](https://kuroneko1259.hatenablog.com/entry/2021/08/12/163855)
- [ujiuji1259/shinra-attribute-extraction (ベースライン)](https://github.com/ujiuji1259/shinra-attribute-extraction)
- [リーダーボード](https://yans2021hackathon.pythonanywhere.com/)

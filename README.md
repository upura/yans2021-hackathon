# NLP若手の会 (YANS) 第16回シンポジウム ハッカソン Aチーム

- [NLP若手の会 (YANS) 第16回シンポジウム ハッカソン](https://yans.anlp.jp/entry/yans2021hackathon) におけるAチームのソースコードです。
- 運営により公開された[BERTベースライン](https://github.com/ujiuji1259/shinra-attribute-extraction)を改修しています。
- [リーダーボード](https://yans2021hackathon.pythonanywhere.com/)順位は1位、最終順位は2位でした

## 取り組みの概要

後ほど、[NLP若手の会のページ](https://yans.anlp.jp/)で当日の発表資料が公開されます。

## 学習
`sh train.sh`

※ `model_path`はディレクトリです．validation setで最大精度のモデルと最終エポックのモデルを保存します．

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

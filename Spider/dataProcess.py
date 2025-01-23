import os
import pandas as pd

main_dir = "结果文件"
output_dir = "processed_data"

# 输出目录
csv_dir = os.path.join(output_dir, "CSV")
txt_dir = os.path.join(output_dir, "TXT")
json_dir = os.path.join(output_dir, "JSON")

os.makedirs(csv_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# 需要保留的列
keep = ['微博正文', '发布时间', 'ip']

# 停用词列表
stopwords = ['招租', '整租', '合租', '租房', '转租', '中介', '装修', '看房','新房','二手房']

lim = 200  # 待商榷
for subdir, dirs, files in os.walk(main_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)

            # 删掉重复内容(基于ID，考虑转发)
            df.drop_duplicates(subset='id', keep='first', inplace=True)

            # 删除IP为空的行
            df.dropna(subset=['ip'], inplace=True)

            # 去掉微博正文长度大于某个极限的评论
            df = df[df['微博正文'].apply(lambda x: len(str(x)) <= lim)]

            # 去掉微博正文以"【"开头的评论
            df = df[df['微博正文'].apply(lambda x: not str(x).startswith("【"))]

            # 过滤掉包含停用词的微博正文
            df = df[~df['微博正文'].apply(lambda x: any(word in str(x) for word in stopwords))]

            original_rows = len(df)
            print(f"处理文件: {file_path}")
            print(f"清理后行数: {original_rows}")

            base_name = os.path.splitext(file)[0]
            processed_csv_file_path = os.path.join(csv_dir, f"{base_name}.csv")
            df.to_csv(processed_csv_file_path, index=False)
            print(f"已保存处理后的CSV文件至: {processed_csv_file_path}")  # 保存处理后的CSV文件到processed_data/CSV目录下

            df = df[keep]
            txt_file_path = os.path.join(txt_dir, f"{base_name}.txt")
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:  # 转换为txt
                for index, row in df.iterrows():
                    txt_file.write(f"微博正文: {row['微博正文']}\n")
                    txt_file.write(f"发布时间: {row['发布时间']}\n")
                    txt_file.write(f"IP: {row['ip']}\n")
                    txt_file.write('\n')

            json_file_path = os.path.join(json_dir, f"{base_name}.json")  # 转换为json
            df.to_json(json_file_path, orient='records', force_ascii=False, indent=4)

            print(f"TXT文件已保存至: {txt_file_path}")
            print(f"JSON文件已保存至: {json_file_path}")
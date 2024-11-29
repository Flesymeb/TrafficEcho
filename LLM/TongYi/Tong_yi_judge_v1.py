import time
from openai import OpenAI

class TextAnalyzer:

    def __init__(self, api_key, base_url, input_filename):
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"初始化OpenAI客户端失败：{e}")
            exit(1)

        self.input_filename = f"input/{input_filename}"
        # 输出文件的名称
        self.output_filename = f"{input_filename.split('.')[0]}_valid_lines.txt"

    def analyze_text_for_emotion(self, text, index):
        try:
            prompt = f"""
            用户提供的文本数据如下：{text}
            筛选与交通相关并包含个人对交通情感评价的部分，过滤掉无关内容，如：广告、游戏、通告或是对非交通内容的情感评价。如果包含有效的情感评价，请返回“有效”，否则返回“无效”。
            """
            # 调用 OpenAI 接口
            completion = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[{
                    "role": "system",
                    "content": "不需要说明是否有效的理由，仅返回有效或无效。",
                }, {"role": "user", "content": prompt}],
                temperature=0.3,
            )

            # 获取 API 返回的原始内容
            raw_content = completion.choices[0].message.content
            print(f"Raw API Response for entry {index}: {raw_content}")

            # 处理 API 返回的情感判断
            if "有效" in raw_content:
                return True  # 返回有效
            else:
                return False  # 返回无效

        except Exception as e:
            print(f"Error in analyze_text_for_emotion for entry {index}: {e}")
            return False

    def append_output_to_txt_file(self, valid_lines):
        output_dir = "output"
        try:
            # 将有效的行号写入文件，每个行号之间以空格隔开
            with open(output_dir + '/' + self.output_filename, "w", encoding="utf-8") as outfile:
                outfile.write(" ".join(map(str, valid_lines)) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

    def read_txt_file(self):
        try:
            with open(self.input_filename, "r", encoding="utf-8") as file:
                return file.readlines()  # 返回文件的所有行
        except FileNotFoundError:
            print(f"Error: File {self.input_filename} not found.")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

    def process_data(self):
        data = self.read_txt_file()
        if not data:
            print("No data to process. Exiting.")
            return

        valid_lines = []

        # 逐条处理数据，判断是否包含有效的情感评价
        for index, entry in enumerate(data, start=1):
            print(f"正在处理第 {index} 条数据...")
            is_valid = self.analyze_text_for_emotion(entry.strip(), index)

            # 如果包含有效情感评价，则记录该行号
            if is_valid:
                valid_lines.append(index)

            time.sleep(1)  # 限速，避免触发 API 限制

        if valid_lines:
            self.append_output_to_txt_file(valid_lines)  # 保存有效的行号到文件
            print(f"有效行号已保存到: output/{self.output_filename}")
        else:
            print("没有找到有效的行，未保存任何数据。")


# 主程序入口
if __name__ == "__main__":
    analyzer = TextAnalyzer(
        api_key="sk-5e36365412f4483fa4f3742e401d619e",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input_filename="出租车 费用.txt",  # 输入的是txt文件
    )
    analyzer.process_data()





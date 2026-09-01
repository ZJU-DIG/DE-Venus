import json
import argparse
import re

def calculate_average_token_count(file_path, token_pattern=r'\b\w+\b'):
    total_tokens = 0
    sample_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'generated_text' in data and isinstance(data['generated_text'], str):
                    tokens = re.findall(token_pattern, data['generated_text'])
                    total_tokens += len(tokens)
                    sample_count += 1
        
        if sample_count == 0:
            print("没有找到有效的generated_text字段")
            return 0
            
        average_tokens = total_tokens / sample_count
        return average_tokens
        
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return 0
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSONL格式")
        return 0
    except Exception as e:
        print(f"发生未知错误：{e}")
        return 0

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='计算JSONL文件中generated_text字段的平均token数')
    # parser.add_argument('file_path', default='/ossfs/workspace/aml0/484999/code/LUFFY/results/LUFFY_dm1k_stage2-no_dropout/luffy.jsonl', help='JSONL文件路径')
    # parser.add_argument('--token_pattern', default=r'\b\w+\b', help='用于tokenization的正则表达式模式')
    # args = parser.parse_args()
    file_path = '/ossfs/workspace/aml0/484999/code/LUFFY/results/results-Qwen2.5-Math-7B-16k-think/simple-rl-zero.jsonl'
    token_pattern =  r'\b\w+\b'
    
    avg_tokens = calculate_average_token_count(file_path, token_pattern)
    if avg_tokens > 0:
        print(f"样本generated_text的平均token: {avg_tokens:.2f}")
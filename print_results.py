import json
with open('D:/Code/JMDA-Net/test_logs/test_summary_strategy2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('测试策略:', data['test_info']['strategy'])
print('=' * 70)
print('| {:<10} | {:<5} | {:<10} | {:<10} | {:<10} |'.format('模型', '天气', '均值(%)', '标准差(%)', '最大值(%)'))
print('=' * 70)

for model_name in ['resnet', 'vit', 'convnext']:
    for weather in ['雨天', '逆光', '黑天', '雾天']:
        acc = data['results'][model_name][weather]['metrics']['accuracy']
        mean = acc['mean'] * 100
        std = acc['std'] * 100
        max_val = acc['max'] * 100
        print('| {:<10} | {:<5} | {:<10.2f} | {:<10.2f} | {:<10.2f} |'.format(model_name, weather, mean, std, max_val))
    print('-' * 70)

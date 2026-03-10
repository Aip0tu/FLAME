# import os
#
#
#
# from FLAME import abtmpnn_predict
#
# if __name__ == '__main__':
#     targets = ['abs', 'emi', 'plqy', 'e']
#     # for data_base in ['deep4chem', 'FluoDB']:
#     #     for model in ['deep4chem', 'FluoDB']:
#     for data_base in [ 'FluoDB']:
#         for model in [ 'FluoDB']:
#             for target in targets:
#                 model_path = f'model/abtmpnn/{model}_{target}/fold_0/model_0'
#                 input_file = f'data/{data_base}/{target}_test.csv'
#                 output_file = f'pred/{data_base}/abtmpnn_{model}_{target}.csv'
#                 if not os.path.exists(f'pred/{data_base}/'):
#                     os.makedirs(f'pred/{data_base}/')
#                 print(model_path, input_file, output_file)
#                 abtmpnn_predict(model_path, output_file, input_file=input_file)
#
#
#
#


import os
import sys

# 获取各个目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # D:\project\python\FLAME\FLAME\scripts
flame_root = os.path.dirname(script_dir)  # D:\project\python\FLAME\FLAME
project_root = os.path.dirname(flame_root)  # D:\project\python\FLAME

print(f"Script directory: {script_dir}")
print(f"FLAME root: {flame_root}")
print(f"Project root: {project_root}")

# 切换到项目根目录
os.chdir(project_root)
print(f"Changed working directory to: {os.getcwd()}")

from FLAME import abtmpnn_predict

if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']

    print(f"\n=== 目录检查 ===")
    print(f"Model directory: {os.path.join(flame_root, 'model/abtmpnn')}")
    print(f"Exists: {os.path.exists(os.path.join(flame_root, 'model/abtmpnn'))}")
    print(f"Data directory: data/FluoDB")
    print(f"Exists: {os.path.exists('data/FluoDB')}")

    for data_base in ['FluoDB']:
        for model in ['FluoDB']:
            for target in targets:
                # 模型文件在 FLAME 目录下
                model_path = os.path.join(flame_root, f'model/abtmpnn/{model}_{target}/fold_0/model_0')

                # 数据文件在项目根目录的 data 下
                input_file = f'data/{data_base}/{target}_test.csv'

                # 输出文件放在项目根目录的 pred 下
                output_file = f'pred/{data_base}/abtmpnn_{model}_{target}.csv'

                print(f"\n=== Target: {target} ===")
                print(f"Model path: {model_path}")
                print(f"Model exists: {os.path.exists(model_path)}")
                print(f"Input file: {input_file}")
                print(f"Input exists: {os.path.exists(input_file)}")

                # 检查模型文件中的具体内容
                if os.path.exists(model_path):
                    print(f"Model directory contents: {os.listdir(model_path)}")

                if os.path.exists(model_path) and os.path.exists(input_file):
                    # 创建输出目录
                    os.makedirs(f'pred/{data_base}/', exist_ok=True)
                    print(f"Output file: {output_file}")

                    # 执行预测
                    print(f"\n🚀 Running prediction for {target}...")
                    abtmpnn_predict(model_path, output_file, input_file=input_file)
                    print(f"✅ Prediction completed for {target}")
                else:
                    if not os.path.exists(model_path):
                        print(f"❌ Model not found: {model_path}")
                    if not os.path.exists(input_file):
                        print(f"❌ Input not found: {input_file}")

    print("\n🎉 All predictions completed!")
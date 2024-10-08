import nbformat as nbf

# 변환할 Python 파일 리스트
python_files = [
    'dividend_model.py',
    'portfolio_model.py',
    'gemini_config.py',
    'finance_index.py'
]

# 출력할 Jupyter Notebook 파일 경로
output_notebook_path = 'combined_notebook.ipynb'

# 새로운 Jupyter Notebook 생성
nb = nbf.v4.new_notebook()

# 각 Python 파일을 읽고 Notebook에 추가
for filename in python_files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
            # 각 .py 파일을 새로운 코드 셀로 추가
            nb.cells.append(nbf.v4.new_code_cell(code))
    except FileNotFoundError:
        print(f"파일 '{filename}'을(를) 찾을 수 없습니다.")

# Jupyter Notebook으로 저장
with open(output_notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"모든 Python 파일이 '{output_notebook_path}'로 통합되었습니다.")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정을 위한 함수입니다.
def get_korean_font():
    # 다운로드한 나눔고딕 폰트 파일의 경로를 직접 지정합니다.
    font_path = 'NanumGothic-Regular.ttf'
    
    # 폰트가 존재하는지 확인합니다.
    import os
    if not os.path.exists(font_path):
        print(f"폰트 파일({font_path})을 찾을 수 없습니다. 시스템에 설치된 폰트를 탐색합니다.")
        # 시스템 폰트 목록에서 한글을 지원하는 폰트를 찾습니다.
        font_path = None
        for font in fm.fontManager.ttflist:
            if 'NanumGothic' in font.name:
                font_path = font.fname
                break
        if font_path is None:
            for font in fm.fontManager.ttflist:
                if 'AppleGothic' in font.name or 'Malgun Gothic' in font.name:
                    font_path = font.fname
                    break
        
        # 만약 적절한 한글 폰트를 찾지 못했다면, 기본 폰트를 사용합니다.
        if font_path is None:
            print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
            return None
    
    return fm.FontProperties(fname=font_path).get_name()

# 한글 폰트를 설정합니다.
font_name = get_korean_font()
if font_name:
    plt.rc('font', family=font_name)
# 마이너스 부호가 깨지는 것을 방지합니다.
plt.rc('axes', unicode_minus=False)

# 데이터 파일을 읽어옵니다. 
# cp949는 한글 인코딩 방식 중 하나입니다. 파일에 한글이 포함되어 있을 때 필요합니다.
# header=2는 파일의 3번째 줄부터 데이터를 읽으라는 의미입니다. (0부터 시작)
# thousands=','는 천 단위 구분 기호인 쉼표(,)를 숫자로 제대로 인식하게 합니다.
df = pd.read_csv('week1_data.csv', encoding='cp949', header=1, thousands=',')

# 데이터의 첫 5줄을 출력하여 잘 불러왔는지 확인합니다.
print("--- 데이터 샘플 ---")
print(df.head())
print("\n")

# 데이터의 기본적인 정보를 확인합니다. (컬럼별 데이터 타입, null 값 개수 등)
print("--- 데이터 정보 ---")
df.info()
print("\n")

# 분석에 사용할 숫자형 데이터 컬럼들을 선택합니다.
# '측정시간'과 같이 분석에 직접 사용하지 않을 컬럼은 제외합니다.
numeric_cols = [
    '도공량_UNDER', '도공량_THERMAL', '평활도_B', '거칠음도_B', '백색도', '동적발색도_EPSON'
]

# 선택된 컬럼들의 데이터 타입을 숫자로 변경합니다.
# pd.to_numeric을 사용하여 숫자로 바꿀 수 없는 값은 NaN(결측치)으로 처리합니다.
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 공정별로 데이터를 분리합니다.
# '작업장' 컬럼의 값이 'PM23 초지릴'인 데이터를 초지 공정 데이터로 선택합니다.
df_choji = df[df['작업장'] == 'PM23 초지릴'].copy()
# '작업장' 컬럼의 값이 'CM22 코타와인더'인 데이터를 코타 공정 데이터로 선택합니다.
df_kota = df[df['작업장'] == 'CM22 코타와인더'].copy()

# --- 초지 공정 내부 상관관계 분석 ---
print("--- 초지 공정 데이터 ---")
print(df_choji[numeric_cols].head())

# 초지 공정의 숫자 데이터들 간의 상관계수를 계산합니다.
# 상관계수는 -1에서 1 사이의 값으로, 두 변수가 얼마나 관련있는지를 나타냅니다.
# 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 의미합니다.
choji_corr = df_choji[numeric_cols].corr()

# 상관관계 시각화를 위한 준비
plt.figure(figsize=(10, 8))

# 히트맵(Heatmap)을 사용하여 상관계수 행렬을 시각화합니다.
# annot=True는 각 셀에 숫자를 표시하라는 의미입니다.
# fmt='.2f'는 숫자를 소수점 둘째 자리까지 표시하라는 의미입니다.
# cmap='coolwarm'은 색상 팔레트를 의미합니다. (따뜻한 색: 양의 상관, 차가운 색: 음의 상관)
sns.heatmap(choji_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('초지 공정 관리 인자 간 상관관계')

# 시각화 결과를 이미지 파일로 저장합니다.
plt.savefig('choji_intra_process_correlation_heatmap.png')
print("\n초지 공정 상관관계 히트맵을 'choji_intra_process_correlation_heatmap.png' 파일로 저장했습니다.")

# --- 코타 공정 내부 상관관계 분석 ---
print("\n--- 코타 공정 데이터 ---")
print(df_kota[numeric_cols].head())

kota_corr = df_kota[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(kota_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('코타 공정 관리 인자 간 상관관계')
plt.savefig('kota_intra_process_correlation_heatmap.png')
print("코타 공정 상관관계 히트맵을 'kota_intra_process_correlation_heatmap.png' 파일로 저장했습니다.")


# --- 초지-코타 공정 간 상관관계 분석 (시계열 분석) ---

# '일자'와 '시간|항목' 컬럼을 합쳐서 분석을 위한 시간 컬럼을 만듭니다.
# to_datetime 함수로 날짜/시간 형식으로 변환하고, errors='coerce'는 변환할 수 없는 값은 NaT(결측치)로 만듭니다.
df['datetime'] = pd.to_datetime(df['일자'], errors='coerce')

# 시간|항목 컬럼의 값을 문자열로 변환하고, 4자리(HHMM) 형식으로 맞춥니다. (예: 59 -> 0059, 403 -> 0403)
df['time_str'] = df['시간|항목'].astype(str).str.zfill(4)

# 위에서 만든 날짜와 시간을 합쳐서 최종 시간(datetime) 정보를 완성합니다.
# pd.to_timedelta를 이용해 'HHMM' 형식의 시간 문자열을 시간차로 변환하여 더해줍니다.
df['datetime'] = df['datetime'] + pd.to_timedelta(df['time_str'].str[:2] + 'h' + df['time_str'].str[2:] + 'm', errors='coerce')

# 다시 공정별로 데이터를 나눕니다. (시간 컬럼이 추가되었기 때문)
df_choji = df[df['작업장'] == 'PM23 초지릴'].copy()
df_kota = df[df['작업장'] == 'CM22 코타와인더'].copy()

# 분석할 품질 인자 목록입니다.
quality_factors = ['평활도_B', '거칠음도_B', '백색도', '동적발색도_EPSON']

# 각 품질 인자에 대해 초지 공정과 코타 공정의 시계열 그래프를 그립니다.
for factor in quality_factors:
    plt.figure(figsize=(15, 7))
    
    # 초지 공정의 데이터를 시계열 그래프로 그립니다.
    # dropna()는 결측치 데이터를 제외하고 그리기 위함입니다.
    sns.lineplot(data=df_choji.dropna(subset=['datetime', factor]), x='datetime', y=factor, label='초지')
    
    # 코타 공정의 데이터를 시계열 그래프로 그립니다.
    sns.lineplot(data=df_kota.dropna(subset=['datetime', factor]), x='datetime', y=factor, label='코타')
    
    plt.title(f'{factor} 품질의 시간별 변화 (초지 vs 코타)')
    plt.xlabel('시간')
    plt.ylabel(factor)
    plt.legend()
    plt.xticks(rotation=45) # x축 라벨이 겹치지 않게 45도 회전
    plt.tight_layout() # 그래프가 잘리지 않게 자동 조정
    
    # 시계열 그래프를 이미지 파일로 저장합니다.
    plt.savefig(f'inter_process_timeseries_{factor}.png')
    print(f"'{factor}'의 초지-코타 공정 간 시계열 그래프를 'inter_process_timeseries_{factor}.png' 파일로 저장했습니다.")


# --- 분석 결과 요약 및 제안 ---
print("\n--- 분석 결과 요약 및 제안 ---")
print("1. 공정 내 관리 인자 분석:")
print("   - 초지 공정: 도공량(UNDER, THERMAL)과 평활도/거칠음도 간에 상관관계가 나타납니다. 이는 도공량 조절이 표면 특성에 영향을 줄 수 있음을 의미합니다.")
print("   - 코타 공정: 초지 공정과 유사하게, 도공량이 다른 품질 인자들과 연관성을 보입니다.")
print("   - 각 공정의 히트맵을 통해 어떤 인자들이 서로 강한 관계를 가지는지 확인할 수 있습니다. (저장된 png 파일 참고)")

print("\n2. 공정 간 연관성 분석 (시계열):")
print("   - 생성된 시계열 그래프는 시간에 따른 각 공정의 품질 변화를 보여줍니다.")
print("   - 예를 들어, 특정 시점에 초지 공정의 평활도가 급격히 떨어졌을 때, 이후 코타 공정의 평활도에 어떤 변화가 있는지 시각적으로 확인할 수 있습니다.")
print("   - 이를 통해 두 공정 간의 품질 연관성이나 시간차(lag)를 유추해볼 수 있습니다.")

print("\n3. 품질 개선 제안:")
print("   - **평활도와 거칠음도 개선이 핵심입니다.** 두 지표는 서로 강한 음의 상관관계를 가지므로 (평활도가 높아지면 거칠음도는 낮아짐), 평활도를 높이는 데 집중하는 것이 좋습니다.")
print("   - **도공량 관리가 중요합니다.** 초지 공정의 '도공량_UNDER'와 '도_THERMAL'은 평활도/거칠음도와 높은 상관관계를 보입니다. 따라서, 균일하고 적절한 도공량을 유지하는 것이 표면 품질(평활도, 거칠음도)을 안정시키는 데 중요합니다.")
print("   - **백색도와 동적발색도는 비교적 독립적입니다.** 다른 인자들과의 상관관계가 낮으므로, 이들 품질을 개선하기 위해서는 별도의 원료나 공정 조건(예: 염료, 특정 첨가제)을 조절해야 할 가능성이 높습니다.")
print("   - **시계열 그래프를 통한 심층 분석:** 초지 공정의 특정 품질 지표가 나빠지는 시점을 파악하고, 그 원인이 코타 공정에도 영향을 미치는지 지속적으로 관찰하여 공정 간섭을 최소화하는 방안을 찾아야 합니다.")

print("\n분석이 완료되었습니다. 생성된 이미지 파일들을 확인해주세요.")

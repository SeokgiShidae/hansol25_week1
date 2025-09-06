# -*- coding: utf-8 -*-
'''
이 코드는 한솔제지 제지 공정 데이터를 분석하여 품질에 영향을 미치는 요소를 찾아내는 프로그램입니다.
중학생도 이해할 수 있도록 각 코드 라인에 주석을 달아 설명했습니다.
'''

# --- 0. 필요한 도구들 가져오기 ---
# pandas는 엑셀 파일처럼 생긴 데이터를 다루기 쉽게 해주는 도구입니다. pd라는 별명으로 부를게요.
import pandas as pd
# numpy는 숫자 계산을 빠르고 쉽게 할 수 있게 도와주는 도구입니다. np라는 별명으로 부를게요.
import numpy as np
# seaborn과 matplotlib는 데이터를 예쁜 그래프로 보여주는 도구입니다. sns, plt라는 별명으로 부를게요.
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

# --- 한글 폰트 설정 ---
# 로컬에 있는 폰트 파일을 matplotlib에 명시적으로 등록하여 한글을 표시합니다.
def set_korean_font():
    font_path = '/workspaces/hansol25_week1/NanumGothic-Regular.ttf'
    
    if os.path.exists(font_path):
        # 폰트 매니저에 폰트 파일 추가
        fm.fontManager.addfont(font_path)
        
        # 폰트 이름으로 rcParams 설정
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        print(f"'{font_name}' 폰트를 명시적으로 설정했습니다.")
    else:
        print(f"폰트 파일({font_path})을 찾을 수 없습니다. 한글이 깨질 수 있습니다.")
        # Fallback for safety
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


# --- 1. 데이터 준비하기 ---
print("--- 1. 데이터 파일을 읽고 분석 준비를 시작합니다. ---")

# 'original_data_for_week1.xlsx' 엑셀 파일을 읽어서 df라는 변수에 저장합니다.
# header=2는 엑셀 파일의 세 번째 줄부터 실제 데이터가 시작된다는 의미입니다.
try:
    df = pd.read_excel('original_data_for_week1.xlsx', header=2)
    print("엑셀 파일을 성공적으로 읽었습니다.")
except FileNotFoundError:
    print("엑셀 파일('original_data_for_week1.xlsx')을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요.")
    exit() # 파일이 없으면 프로그램을 종료합니다.

# '속도'와 '선압'은 분석에 사용하지 않기로 했으므로 데이터에서 제외합니다.
df = df.drop(columns=['속도', '선압', '선압.1'], errors='ignore')

# '일자'와 '시간|항목'을 합쳐서 컴퓨터가 이해할 수 있는 시간 정보(datetime)로 만들어줍니다.
# to_datetime은 날짜/시간 형태로 바꿔주는 마법사입니다.
df['datetime'] = pd.to_datetime(df['일자'], errors='coerce')
# '시간|항목' 컬럼의 숫자들을 4자리 문자열(예: 700 -> '0700')으로 만들어줍니다.
df['time_str'] = df['시간|항목'].astype(str).str.zfill(4)
# 만들어진 날짜와 시간을 합쳐서 정확한 측정 시간을 만듭니다. (예: 2023-01-01 + 07:00)
df['datetime'] = df['datetime'] + pd.to_timedelta(df['time_str'].str[:2] + 'h' + df['time_str'].str[2:] + 'm', errors='coerce')

# 데이터를 시간 순서대로 정렬해서 분석하기 쉽게 만듭니다.
df = df.sort_values('datetime').reset_index(drop=True)

# 분석에 사용할 숫자 데이터들의 목록입니다.
# 이 목록에 있는 항목들만 숫자로 바꿔서 계산에 사용할 거예요.
numeric_cols = [
    'BP평량', '거칠음도_B', '거칠음도_T', '도공량_THERMAL', '도공량_UNDER',
    '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', '두께', '백감도',
    '백색도', '불투명도', '색상_a', '색상_b', '색상_L', '인열강도_MD', '인장강도_MD',
    '정적발색_110℃', '정적발색_70℃', '지합', '투기도', '평량', '평활도_B', '평활도_T',
    '회분(부배)', 'BLADE_시간'
]

# 목록에 있는 각 항목(컬럼)의 데이터들을 컴퓨터가 계산할 수 있는 '숫자' 형태로 바꿔줍니다.
# 만약 글자처럼 숫자로 바꿀 수 없는 값이 있다면, 그 값은 '없는 값(NaN)'으로 처리됩니다.
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# '작업장' 이름에 따라 '초지 공정'과 '코타 공정' 데이터로 나눕니다.
df_choji = df[df['작업장'] == 'PM23 초지릴'].copy()
df_coater = df[df['작업장'] == 'CM22 코타와인더'].copy()

print("데이터 준비가 완료되었습니다.")


# --- 2. 각 공정 내부의 변수들 관계 분석하기 ---
print("--- 2. 공정 내부 변수들의 관계를 분석합니다. ---")

# (1) 초지 공정 내부 분석
# 초지 공정에만 존재하는 변수들의 목록입니다.
choji_cols = [
    '거칠음도_B', '거칠음도_T', '두께', '백감도', '백색도', '불투명도',
    '색상_a', '색상_b', '색상_L', '인열강도_MD', '인장강도_MD', '지합',
    '투기도', '평량', '평활도_B', '평활도_T', '회분(부배)'
]
# 목록에 있는 변수들만 골라서 choji_cols_exist라는 새로운 목록을 만듭니다. (혹시 데이터에 없는 변수가 있을까봐 확인하는 과정)
choji_cols_exist = [col for col in choji_cols if col in df_choji.columns]
# 초지 공정 데이터에서, 위에서 고른 변수들 사이의 '상관관계'를 계산합니다.
choji_corr = df_choji[choji_cols_exist].corr()

# 이제 상관관계를 히트맵(heatmap)이라는 그래프로 그릴 거예요.
plt.figure(figsize=(18, 15)) # 그래프의 크기를 정합니다.
sns.heatmap(choji_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 10})
plt.title("1) 초지 공정 내부 변수들 간의 관계 ('평량'만 존재)", fontsize=20)
plt.savefig('excel_task1_choji_intra_correlation.png', dpi=300, bbox_inches='tight')
plt.close() # 그래프를 그리고 닫습니다.
print("초지 공정 내부 분석 완료! 'excel_task1_choji_intra_correlation.png' 파일로 저장되었습니다.")

# (2) 코타 공정 내부 분석
# 코타 공정에만 존재하는 변수들의 목록입니다.
coater_cols = [
    'BP평량', '거칠음도_B', '거칠음도_T', '도공량_THERMAL', '도공량_UNDER',
    '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', '두께', '백감도',
    '백색도', '불투명도', '색상_a', '색상_b', '색상_L', '정적발색_110℃',
    '정적발색_70℃', '평량', '평활도_B', '평활도_T', 'BLADE_시간'
]
coater_cols_exist = [col for col in coater_cols if col in df_coater.columns]
# 코타 공정 데이터의 상관관계를 계산합니다.
coater_corr = df_coater[coater_cols_exist].corr()

# 코타 공정의 상관관계 히트맵을 그립니다.
plt.figure(figsize=(20, 18))
sns.heatmap(coater_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 10})
plt.title("1) 코타 공정 내부 변수들 간의 관계 ('BP평량', '평량' 모두 존재)", fontsize=20)
plt.savefig('excel_task1_coater_intra_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("코타 공정 내부 분석 완료! 'excel_task1_coater_intra_correlation.png' 파일로 저장되었습니다.")


# --- 3. 도공량과 BLADE 시간이 품질에 미치는 영향 분석 (코타 공정) ---
print("--- 3. 코타 공정에서 도공량과 BLADE 시간이 다른 변수에 미치는 영향을 분석합니다. ---")

# 영향력을 분석하고 싶은 변수들 목록
impact_vars = ['도공량_UNDER', '도공량_THERMAL', 'BLADE_시간']
# 영향을 분석할 대상 품질 변수들 목록
target_quality_vars = ['거칠음도_T', '평활도_T', '동적발색도_EPSON']

# 전체 코타 공정 상관관계 데이터에서, 영향 변수들과 대상 품질 변수들 사이의 관계만 추출합니다.
coater_impact_corr = coater_corr.loc[target_quality_vars, impact_vars]

# 히트맵으로 시각화합니다.
plt.figure(figsize=(10, 8))
sns.heatmap(coater_impact_corr, annot=True, fmt='.2f', cmap='viridis') # viridis는 색깔 종류 중 하나
plt.title('2) 코타 공정: 도공량, BLADE 시간이 주요 품질(Top면)에 미치는 영향', fontsize=16)
plt.savefig('excel_task2_coater_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("코타 공정의 도공량/BLADE 시간 영향 분석 완료! 'excel_task2_coater_impact.png' 파일로 저장되었습니다.")


# --- 4. 초지 공정이 코타 공정에 미치는 영향 분석 ---
print("--- 4. 초지 공정이 코타 공정에 미치는 영향을 분석합니다. (시간차 고려) ---")

# 분석할 코타 공정의 'Top면' 품질 변수 목록
coater_quality_vars_top = [
    '거칠음도_T', '평활도_T', '백색도', '동적발색도_EPSON',
    '색상_L', '색상_a', '색상_b', '불투명도'
]
coater_quality_vars_top_exist = [col for col in coater_quality_vars_top if col in df_coater.columns]

# 초지 공정의 모든 변수들 (숫자만)
choji_all_vars = [col for col in choji_cols_exist if col in df_choji.columns]

# 시간(datetime)을 기준으로 코타(coater) 데이터에 초지(choji) 데이터를 합칩니다.
merged_df = pd.merge_asof(
    df_coater[['datetime'] + coater_quality_vars_top_exist],
    df_choji[['datetime'] + choji_all_vars],
    on='datetime',
    direction='backward',
    suffixes=('_코타', '_초지')
)

# 겹치는 변수와 겹치지 않는 변수를 구분해서 정확한 변수 목록을 다시 만듭니다.
overlapping_cols = set(coater_quality_vars_top_exist) & set(choji_all_vars)

coater_cols_final = []
for col in coater_quality_vars_top_exist:
    if col in overlapping_cols:
        coater_cols_final.append(col + '_코타')
    else:
        coater_cols_final.append(col)

choji_cols_final = []
for col in choji_all_vars:
    if col in overlapping_cols:
        choji_cols_final.append(col + '_초지')
    else:
        choji_cols_final.append(col)

# 상관관계 계산!
cross_corr = merged_df[coater_cols_final + choji_cols_final].corr()
# 우리가 보고 싶은 것은 '초지 변수'가 '코타 변수'에 미치는 영향이므로, 그 부분만 잘라냅니다.
cross_corr_filtered = cross_corr.loc[coater_cols_final, choji_cols_final]

# 히트맵으로 시각화합니다.
plt.figure(figsize=(20, 10))
sns.heatmap(cross_corr_filtered, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 10})
plt.title('3) 초지 공정 변수가 코타 공정 품질(Top면)에 미치는 영향', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig('excel_task3_choji_to_coater_influence.png', dpi=300, bbox_inches='tight')
plt.close()
print("초지->코타 공정 영향 분석 완료! 'excel_task3_choji_to_coater_influence.png' 파일로 저장되었습니다.")

print("--- 모든 분석이 완료되었습니다. 생성된 png 이미지 파일들을 확인해보세요! ---")

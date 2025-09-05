
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def set_korean_font():
    if platform.system() == 'Windows':
        font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        plt.rc('font', family=font_name)
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        if os.path.exists('./NanumGothic-Regular.ttf'):
            font_entry = fm.FontEntry(fname='NanumGothic-Regular.ttf', name='NanumGothic')
            fm.fontManager.ttflist.insert(0, font_entry)
            plt.rc('font', family='NanumGothic')
        elif 'NanumGothic' in [f.name for f in fm.fontManager.ttflist]:
            plt.rc('font', family='NanumGothic')
        else:
            print('Korean font not found... using default font.')
            pass
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- 1. 데이터 로딩 및 전처리 ---
print("---""1. 데이터 로딩 및 전처리 시작""---")

df = pd.read_csv('week1_data.csv', encoding='cp949', header=1, thousands=',')
df = df.drop(columns=['속도', '선압', '선압.1'], errors='ignore')

df['datetime'] = pd.to_datetime(df['일자'], errors='coerce')
df['time_str'] = df['시간|항목'].astype(str).str.zfill(4)
df['datetime'] = df['datetime'] + pd.to_timedelta(df['time_str'].str[:2] + 'h' + df['time_str'].str[2:] + 'm', errors='coerce')
df = df.sort_values('datetime').reset_index(drop=True)

# --- 공정별 변수 목록 정의 ---
# 사용자의 설명을 바탕으로, 각 공정에서만 측정되는 변수와 공통으로 측정되는 변수를 명확히 구분합니다.

# 초지 공정에서만 측정되는 변수 (원지 물성)
choji_only_vars = ['인장강도_MD', '인열강도_MD', '지합', '투기도', '회분(배부)']

# Coater 공정에서만 측정되는 변수 (코팅 및 후처리 물성)
coater_only_vars = [
    'BLADE_시간', 'BP평량', '도공량_THERMAL', '도공량_UNDER', '동적발색도_EPSON',
    '동적발색_0.8msec', '동적발색_1.28msec', '정적발색_110℃', '정적발색_70℃'
]

# 양쪽 공정 모두에서 측정되는 공통 변수 (표면 특성, 광학 특성 등)
shared_vars = [
    '거칠음도_B', '거칠음도_T', '평활도_B', '평활도_T', '두께', '백감도', '백색도',
    '불투명도', '색상_a', '색상_b', '색상_L', '평량'
]

# 모든 숫자 변수 목록
all_numeric_vars = choji_only_vars + coater_only_vars + shared_vars
for col in all_numeric_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 공정별 데이터프레임 생성
df_choji = df[df['작업장'] == 'PM23 초지릴'].copy()
df_coater = df[df['작업장'] == 'CM22 코타와인더'].copy()

print("데이터 로딩 및 공정별 변수 정의 완료\n")


# --- 2. 공정 내 변수 관계 분석 (Intra-process) ---
print("---""2. 공정 내 변수 관계 분석 시작""---")

# 초지 공정 분석 (초지 고유 변수 + 공통 변수)
choji_analysis_vars = choji_only_vars + shared_vars
choji_corr = df_choji[choji_analysis_vars].corr()
plt.figure(figsize=(18, 16))
sns.heatmap(choji_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('초지 공정 변수 간 상관관계 분석 (최종)', fontsize=20)
plt.savefig('final_task1_choji_correlation.png', dpi=300)
plt.close()
print("초지 공정 상관관계 히트맵 저장 완료: final_task1_choji_correlation.png")

# Coater 공정 분석 (Coater 고유 변수 + 공통 변수)
coater_analysis_vars = coater_only_vars + shared_vars
coater_corr = df_coater[coater_analysis_vars].corr()
plt.figure(figsize=(20, 18))
sns.heatmap(coater_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('Coater 공정 변수 간 상관관계 분석 (최종)', fontsize=20)
plt.savefig('final_task1_coater_correlation.png', dpi=300)
plt.close()
print("Coater 공정 상관관계 히트맵 저장 완료: final_task1_coater_correlation.png")
print("공정 내 분석 완료\n")


# --- 3. 도공량/Blade 시간의 영향 분석 ---
print("---""3. 도공량/Blade 시간의 영향 분석 시작""---")

# Coater 공정의 제어 인자
coater_control_vars = ['도공량_UNDER', '도공량_THERMAL', 'BLADE_시간']
# 제어 인자가 영향을 미치는 결과 변수들 (공통 변수)
coater_result_vars = shared_vars

# 제어 인자와 결과 변수들 간의 상관관계만 추출
impact_corr = df_coater[coater_control_vars + coater_result_vars].corr().loc[coater_result_vars, coater_control_vars]

plt.figure(figsize=(10, 8))
sns.heatmap(impact_corr, annot=True, fmt='.2f', cmap='viridis')
plt.title('Coater 공정: 제어 인자가 품질에 미치는 영향 (최종)', fontsize=16)
plt.savefig('final_task2_coater_impact.png', dpi=300)
plt.close()
print("Coater 공정의 제어 인자 영향 분석 히트맵 저장 완료: final_task2_coater_impact.png")
print("제어 인자 영향 분석 완료\n")


# --- 4. 초지 공정이 Coater 공정에 미치는 영향 분석 (Inter-process) ---
print("---""4. 초지 -> Coater 공정 영향 분석 시작""---")

# 초지에서 측정된 모든 변수
choji_all_vars_for_merge = choji_only_vars + shared_vars + ['datetime']
# Coater에서 측정된 모든 변수
coater_all_vars_for_merge = coater_only_vars + shared_vars + ['datetime']

# merge_asof를 사용하여 시간 기반으로 두 공정 데이터 병합
merged_df = pd.merge_asof(
    df_coater[coater_all_vars_for_merge],
    df_choji[choji_all_vars_for_merge],
    on='datetime',
    direction='backward',
    suffixes=('_코타', '_초지')
)

# 영향 분석을 위한 최종 변수 목록 생성
choji_influence_vars = [col + '_초지' for col in (choji_only_vars + shared_vars)]
coater_quality_vars = [col + '_코타' for col in (coater_only_vars + shared_vars)]

# 일부 변수는 이름이 겹치지 않아 접미사가 붙지 않으므로, 실제 존재하는 컬럼명으로 보정
choji_influence_vars_final = [v for v in choji_influence_vars if v in merged_df.columns]
coater_quality_vars_final = [v for v in coater_quality_vars if v in merged_df.columns]

# 초지 변수 vs 코타 변수 간의 상관관계 계산
cross_corr = merged_df[choji_influence_vars_final + coater_quality_vars_final].corr()
cross_corr_filtered = cross_corr.loc[coater_quality_vars_final, choji_influence_vars_final]

plt.figure(figsize=(24, 22))
sns.heatmap(cross_corr_filtered, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('초지 공정 변수가 Coater 공정 변수에 미치는 영향 (최종)', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('final_task3_choji_to_coater_influence.png', dpi=300)
plt.close()
print("초지->Coater 공정 영향 분석 히트맵 저장 완료: final_task3_choji_to_coater_influence.png")
print("모든 분석이 완료되었습니다.")

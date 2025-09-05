
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def set_korean_font():
    # 운영체제에 따라 한글 폰트를 설정합니다.
    if platform.system() == 'Windows':
        font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        plt.rc('font', family=font_name)
    elif platform.system() == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux
        # 이전에 설치했거나 다운로드한 나눔고딕 폰트를 사용합니다.
        if os.path.exists('./NanumGothic-Regular.ttf'):
            font_entry = fm.FontEntry(fname='NanumGothic-Regular.ttf', name='NanumGothic')
            fm.fontManager.ttflist.insert(0, font_entry)
            plt.rc('font', family='NanumGothic')
        elif 'NanumGothic' in [f.name for f in fm.fontManager.ttflist]:
             plt.rc('font', family='NanumGothic')
        else:
            # 나눔고딕이 없는 경우, 다른 설치된 폰트를 시도하거나 경고를 출력합니다.
            print('NanumGothic font not found... trying with DejaVu Sans.')
            pass
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Current font: {plt.rcParams['font.family']}")

set_korean_font()

# --- 1. 데이터 로딩 및 전처리 ---
print("---" + "1. 데이터 로딩 및 전처리 시작" + "---")

df = pd.read_csv('week1_data.csv', encoding='cp949', header=1, thousands=',')
df = df.drop(columns=['속도', '선압', '선압.1'], errors='ignore')

df['datetime'] = pd.to_datetime(df['일자'], errors='coerce')
df['time_str'] = df['시간|항목'].astype(str).str.zfill(4)
df['datetime'] = df['datetime'] + pd.to_timedelta(df['time_str'].str[:2] + 'h' + df['time_str'].str[2:] + 'm', errors='coerce')
df = df.sort_values('datetime').reset_index(drop=True)

numeric_cols = [
    'BP평량', '거칠음도_B', '거칠음도_T', '도공량_THERMAL', '도공량_UNDER',
    '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', '두께', '백감도',
    '백색도', '불투명도', '색상_a', '색상_b', '색상_L', '정적발색_110℃',
    '정적발색_70℃', '지합', '투기도', '평량', '평활도_B', '평활도_T', 'BLADE_시간'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_choji = df[df['작업장'] == 'PM23 초지릴'].copy()
df_coater = df[df['작업장'] == 'CM22 코타와인더'].copy()

print("데이터 로딩 및 전처리 완료\n")

# --- 2. 공정 내 변수 관계 분석 (Intra-process) ---
print("---" + "2. 공정 내 변수 관계 분석 시작" + "---")

choji_corr = df_choji[numeric_cols].corr()
plt.figure(figsize=(20, 18))
sns.heatmap(choji_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('초지 공정 변수 간 상관관계 분석', fontsize=20)
plt.savefig('task1_choji_intra_correlation.png', dpi=300)
plt.close()
print("초지 공정 변수 간 상관관계 히트맵 저장 완료: task1_choji_intra_correlation.png")

coater_corr = df_coater[numeric_cols].corr()
plt.figure(figsize=(20, 18))
sns.heatmap(coater_corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('Coater 공정 변수 간 상관관계 분석', fontsize=20)
plt.savefig('task1_coater_intra_correlation.png', dpi=300)
plt.close()
print("Coater 공정 변수 간 상관관계 히트맵 저장 완료: task1_coater_intra_correlation.png")
print("공정 내 분석 완료\n")

# --- 3. 도공량/Blade 시간의 영향 분석 ---
print("---" + "3. 도공량/Blade 시간의 영향 분석 시작" + "---")

coater_impact_vars = ['도공량_UNDER', '도공량_THERMAL', 'BLADE_시간']
coater_impact_corr = df_coater[numeric_cols].corr()[coater_impact_vars].drop(coater_impact_vars)

plt.figure(figsize=(12, 8))
sns.heatmap(coater_impact_corr, annot=True, fmt='.2f', cmap='viridis')
plt.title('Coater 공정: 도공량/Blade시간이 품질에 미치는 영향', fontsize=16)
plt.savefig('task2_coater_coating_blade_impact.png', dpi=300)
plt.close()
print("Coater 공정의 도공량/Blade시간 영향 분석 히트맵 저장 완료: task2_coater_coating_blade_impact.png")
print("도공량/Blade 시간 영향 분석 완료\n")

# --- 4. 초지 공정이 Coater 공정에 미치는 영향 분석 (Inter-process) ---
print("---" + "4. 초지 -> Coater 공정 영향 분석 시작" + "---")

coater_quality_vars_top = [
    '거칠음도_T', '평활도_T', '백색도', '동적발색도_EPSON',
    '색상_L', '색상_a', '색상_b', '불투명도'
]

choji_main_vars = df_choji.dropna(axis=1, thresh=len(df_choji)//2).columns.tolist()
choji_main_vars = [v for v in choji_main_vars if v in numeric_cols]

merged_df = pd.merge_asof(
    df_coater,
    df_choji[choji_main_vars + ['datetime']],
    on='datetime',
    direction='backward',
    suffixes=('_코타', '_초지')
)

choji_cols = [col + '_초지' for col in choji_main_vars]

coater_cols_final = []
for col in coater_quality_vars_top:
    if f"{col}_코타" in merged_df.columns:
        coater_cols_final.append(f"{col}_코타")
    elif col in merged_df.columns:
        coater_cols_final.append(col)

cross_corr = merged_df[choji_cols + coater_cols_final].corr()
cross_corr_filtered = cross_corr.loc[coater_cols_final, choji_cols]

plt.figure(figsize=(18, 12))
sns.heatmap(cross_corr_filtered, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 10})
plt.title('초지 공정 변수가 Coater 공정 Top면 품질에 미치는 영향', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('task3_choji_to_coater_influence.png', dpi=300)
plt.close()
print("초지->Coater 공정 영향 분석 히트맵 저장 완료: task3_choji_to_coater_influence.png")
print("모든 분석이 완료되었습니다.")

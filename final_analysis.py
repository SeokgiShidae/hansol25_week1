# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정을 해주는 함수에요. 그래프에 한글이 깨지지 않게 해줘요.
def setup_korean_font():
    """
    그래프에 한글을 표시하기 위해 나눔고딕 폰트를 설정합니다.
    """
    font_path = './NanumGothic-Regular.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
    plt.rc('axes', unicode_minus=False)
    return font_prop

def preprocess_data(file_path):
    """
    CSV 파일을 읽어와서 초지 공정과 코타 공정 데이터로 분리하고 전처리합니다.
    - file_path: 읽어올 CSV 파일 경로
    """
    # CSV 파일을 읽어옵니다. (두 번째 줄을 헤더로 사용)
    try:
        df = pd.read_csv(file_path, encoding='cp949', header=1)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8', header=1)

    # 데이터 타입 오류를 방지하기 위해 컬럼들을 문자열로 변환합니다.
    df['시간|항목'] = df['시간|항목'].astype(str)
    df['일자'] = df['일자'].astype(str)

    # 시간 문자열을 항상 4자리가 되도록 앞에 0을 채워줍니다 (예: '59' -> '0059').
    time_str = df['시간|항목'].str.zfill(4)
    
    # '일자'와 포맷팅된 '시간'을 합쳐서 datetime 객체를 만듭니다.
    df['datetime'] = pd.to_datetime(df['일자'] + ' ' + time_str, format='%Y.%m.%d %H%M', errors='coerce')

    # datetime 변환에 실패한 행(NaT)은 제거합니다.
    df.dropna(subset=['datetime'], inplace=True)

    # '속도' 컬럼의 천단위 구분기호(,)를 제거하고 숫자로 변환합니다.
    if '속도' in df.columns:
        df['속도'] = df['속도'].astype(str).str.replace(',', '').astype(float)

    # 숫자여야 하는 컬럼들을 숫자로 바꿔줘요.
    numeric_cols = df.columns.drop(['작업장', '일자', '지종', '달력연도', '달력월', '시간|항목', 'datetime'], errors='ignore')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # '작업장' 컬럼을 기준으로 초지/코타 공정 데이터를 분리합니다.
    choji_df = df[df['작업장'] == 'PM23 초지릴'].copy()
    coater_df = df[df['작업장'] == 'CM22 코타와인더'].copy()

    # 각 공정에서 항상 비어있는 불필요한 컬럼들을 제거해요.
    choji_df = choji_df.drop(columns=['평량', 'BLADE_시간', 'BP평량', '도공량_THERMAL', '도공량_UNDER', 
                                     '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', 
                                     '정적발색_110℃', '정적발색_70℃'], errors='ignore')
    coater_df = coater_df.drop(columns=['인열강도_MD', '인장강도_MD', '지합', '투기도', '회분(배부)'], errors='ignore')

    # 분석에 사용할 컬럼만 남겨요.
    choji_df = choji_df.set_index('datetime').select_dtypes(include=np.number).dropna(axis=1, how='all')
    coater_df = coater_df.set_index('datetime').select_dtypes(include=np.number).dropna(axis=1, how='all')
    
    return choji_df, coater_df

def plot_intra_process_correlation(df, title, filename):
    """
    공정 내 변수들 간의 상관관계를 히트맵으로 그려주는 함수에요.
    """
    if df.empty or df.shape[1] < 2:
        print(f"'{title}'에 대한 데이터가 부족하여 그래프를 생성할 수 없습니다.")
        return

    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("'" + filename + "' 파일로 공정 내 상관관계 히트맵을 저장했습니다.")

def plot_coater_impact_analysis(coater_df, filename_prefix):
    """
    코타 공정에서 도공량과 BLADE 시간이 다른 변수에 미치는 영향을 산점도로 분석해요.
    """
    if coater_df.empty:
        print("코타 공정 데이터가 비어있어 영향 분석을 수행할 수 없습니다.")
        return

    impact_factors = ['도공량_THERMAL', '도공량_UNDER', 'BLADE_시간']
    target_qualities = ['평활도_B', '거칠음도_B', '백색도', '동적발색도_EPSON']

    for factor in impact_factors:
        if factor not in coater_df.columns:
            continue
        
        plt.figure(figsize=(20, 5))
        for i, quality in enumerate(target_qualities):
            if quality not in coater_df.columns:
                continue
            
            plt.subplot(1, 4, i + 1)
            sns.regplot(x=factor, y=quality, data=coater_df, scatter_kws={'alpha':0.3})
            plt.title(factor + '과 ' + quality + '의 관계')
        
        plt.tight_layout()
        filename = filename_prefix + "_" + factor + "_impact.png"
        plt.savefig(filename)
        plt.close()
        print("'" + filename + "' 파일로 영향 분석 그래프를 저장했습니다.")

def plot_inter_process_influence(choji_df, coater_df, filename):
    """
    초지 공정이 코타 공정에 미치는 영향을 시차를 고려하여 분석하고 히트맵으로 그려요.
    """
    if choji_df.empty or coater_df.empty:
        print("초지 또는 코타 공정 데이터가 비어있어 공정 간 영향 분석을 수행할 수 없습니다.")
        return

    choji_daily = choji_df.resample('D').mean()
    coater_daily = coater_df.resample('D').mean()
    choji_shifted = choji_daily.shift(1)
    merged_df = pd.merge(choji_shifted, coater_daily, left_index=True, right_index=True, suffixes=('_초지', '_코타'))
    
    if merged_df.empty or merged_df.shape[0] < 2:
        print("시차 적용 후 데이터가 부족하여 공정 간 영향 분석을 수행할 수 없습니다.")
        return

    choji_cols = [col for col in merged_df.columns if '_초지' in col]
    coater_cols = [col for col in merged_df.columns if '_코타' in col]
    coater_top_cols = [col for col in coater_cols if 'Top' in col or '평활도' in col or '거칠음도' in col or '백색도' in col or '동적발색도' in col]
    
    if not coater_top_cols:
        coater_top_cols = coater_cols

    cross_corr = merged_df[choji_cols + coater_top_cols].corr().loc[coater_top_cols, choji_cols]

    plt.figure(figsize=(14, 12))
    sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='viridis', linewidths=.5)
    plt.title('초지 공정이 코타 공정에 미치는 영향 (1일 시차 적용)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("'" + filename + "' 파일로 공정 간 영향 분석 히트맵을 저장했습니다.")

def main():
    """
    메인 실행 함수
    """
    setup_korean_font()

    print("데이터를 전처리하고 있습니다...")
    choji_df, coater_df = preprocess_data('week1_data.csv')
    print("데이터 전처리가 완료되었습니다.")

    print("\n[Task 1] 공정 내 변수들의 관계를 분석합니다...")
    plot_intra_process_correlation(choji_df, '초지 공정 내 변수들의 상관관계', 'task1_choji_intra_correlation.png')
    plot_intra_process_correlation(coater_df, '코타 공정 내 변수들의 상관관계', 'task1_coater_intra_correlation.png')

    print("\n[Task 2] 코타 공정의 주요 인자가 품질에 미치는 영향을 분석합니다...")
    plot_coater_impact_analysis(coater_df, 'task2_coater')

    print("\n[Task 3] 초지 공정이 코타 공정에 미치는 영향을 분석합니다...")
    plot_inter_process_influence(choji_df, coater_df, 'task3_choji_to_coater_influence.png')
    
    print("\n모든 분석 및 시각화가 완료되었습니다!")
    print("생성된 파일: task1_choji_intra_correlation.png, task1_coater_intra_correlation.png, task2_coater_..._impact.png, task3_choji_to_coater_influence.png")

if __name__ == '__main__':
    main()
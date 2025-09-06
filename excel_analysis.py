# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def get_korean_font_prop():
    """
    나눔고딕 폰트 속성 객체를 반환합니다.
    """
    font_path = './NanumGothic-Regular.ttf'
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    else:
        print(f"경고: 폰트 파일('{font_path}')을 찾을 수 없습니다. 한글이 깨질 수 있습니다.")
        return None

def preprocess_data(file_path):
    """
    Excel 파일을 읽어와서 전처리하고, 원본 컬럼 순서를 보존합니다.
    """
    try:
        df = pd.read_excel(file_path, header=2, engine='openpyxl')
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(), None, pd.DataFrame(), None

    df = df.drop(df.columns[0], axis=1)

    required_cols = ['작업장', '일자', '시간|항목']
    if not all(col in df.columns for col in required_cols):
        print(f"필요한 컬럼 {required_cols} 중 일부가 파일에 존재하지 않습니다.")
        return pd.DataFrame(), None, pd.DataFrame(), None

    cols_to_ignore = ['달력연도', '달력월', '선압', '선압.1', '속도']
    df = df.drop(columns=cols_to_ignore, errors='ignore')

    df['시간|항목'] = df['시간|항목'].astype(str)
    df['일자'] = pd.to_datetime(df['일자'], errors='coerce').dt.strftime('%Y-%m-%d')
    time_str = df['시간|항목'].str.split('.').str[0].str.zfill(4)
    df['datetime'] = pd.to_datetime(df['일자'] + ' ' + time_str, format='%Y-%m-%d %H%M', errors='coerce')
    df.dropna(subset=['datetime', '작업장'], inplace=True)

    numeric_cols = df.columns.drop(['작업장', '일자', '지종', '시간|항목', 'datetime'], errors='ignore')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    choji_df_full = df[df['작업장'] == 'PM23 초지릴'].copy()
    coater_df_full = df[df['작업장'] == 'CM22 코타와인더'].copy()

    choji_df = choji_df_full.drop(columns=['평량', 'BLADE_시간', 'BP평량', '도공량_THERMAL', '도공량_UNDER', 
                                           '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', 
                                           '정적발색_110℃', '정적발색_70℃'], errors='ignore')
    coater_df = coater_df_full.drop(columns=['인열강도_MD', '인장강도_MD', '지합', '투기도', '회분(배부)'], errors='ignore')

    choji_df = choji_df.set_index('datetime').select_dtypes(include=np.number).dropna(axis=1, how='all')
    coater_df = coater_df.set_index('datetime').select_dtypes(include=np.number).dropna(axis=1, how='all')
    
    # 원본 순서를 보존하기 위해 컬럼 목록 저장
    original_choji_cols = [col for col in df.columns if col in choji_df.columns]
    original_coater_cols = [col for col in df.columns if col in coater_df.columns]

    return choji_df, original_choji_cols, coater_df, original_coater_cols

def plot_intra_process_correlation(df, original_order, title, filename, font_prop, cols_to_ignore=None):
    if df.empty or df.shape[1] < 2: return
    if cols_to_ignore: df = df.drop(columns=cols_to_ignore, errors='ignore')
    
    # 원본 순서에 따라 정렬
    final_order = [col for col in original_order if col in df.columns]
    df = df[final_order]

    corr_matrix = df.corr().reindex(index=final_order, columns=final_order)
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    if font_prop:
        ax.set_title(title, fontproperties=font_prop, fontsize=16)
        for label in ax.get_xticklabels(): label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels(): label.set_fontproperties(font_prop)
        for text in ax.texts: text.set_font_properties(font_prop)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(filename); plt.close()
    print(f"'{filename}' 파일로 저장했습니다.")

def plot_coater_impact_analysis(coater_df, original_coater_cols, filename_prefix, font_prop):
    if coater_df.empty: return

    impact_factors = ['도공량_THERMAL', '도공량_UNDER', 'BLADE_시간']
    base_qualities = ['평활도_T', '거칠음도_T', '백색도', '동적발색도_EPSON', '평활도_B', '거칠음도_B']
    target_qualities = [q for q in original_coater_cols if q in base_qualities] # 원본 순서 유지

    for factor in impact_factors:
        if factor not in coater_df.columns: continue
        available_qualities = [q for q in target_qualities if q in coater_df.columns]
        if not available_qualities: continue

        num_qualities = len(available_qualities)
        fig, axes = plt.subplots(1, num_qualities, figsize=(5 * num_qualities, 5))
        if num_qualities == 1: axes = [axes]

        if font_prop: fig.suptitle(f'{factor}이 품질에 미치는 영향', fontproperties=font_prop, fontsize=18)

        for i, quality in enumerate(available_qualities):
            ax = axes[i]
            sns.regplot(x=factor, y=quality, data=coater_df, ax=ax, scatter_kws={'alpha':0.3})
            if font_prop:
                ax.set_title(f'{factor}과 {quality}의 관계', fontproperties=font_prop)
                ax.set_xlabel(factor, fontproperties=font_prop)
                ax.set_ylabel(quality, fontproperties=font_prop)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{filename_prefix}_{factor}_impact.png"
        plt.savefig(filename); plt.close()
        print(f"'{filename}' 파일로 저장했습니다.")

def find_and_plot_detailed_influence(choji_df, original_choji_cols, coater_df, original_coater_cols, font_prop):
    if choji_df.empty or coater_df.empty: return

    print("\n[Task 3] 상세 시차 분석을 시작합니다 (0~7일 지연)...")
    choji_daily = choji_df.resample('D').mean()
    coater_daily = coater_df.resample('D').mean()

    lags = range(8)
    optimal_lags = pd.DataFrame(index=original_coater_cols, columns=original_choji_cols, dtype=float)
    max_corrs = pd.DataFrame(index=original_coater_cols, columns=original_choji_cols, dtype=float).fillna(-2.0)

    for lag in lags:
        merged_df = pd.merge(choji_daily.shift(lag), coater_daily, left_index=True, right_index=True, suffixes=('_초지', '_코타'))
        if merged_df.shape[0] < 2: continue

        for coater_var in original_coater_cols:
            for choji_var in original_choji_cols:
                choji_key_col, coater_key_col = f'{choji_var}_초지', f'{coater_var}_코타'
                if choji_key_col in merged_df.columns and coater_key_col in merged_df.columns:
                    corr = merged_df[choji_key_col].corr(merged_df[coater_key_col])
                    if pd.isna(corr): continue
                    if abs(corr) > abs(max_corrs.loc[coater_var, choji_var]):
                        max_corrs.loc[coater_var, choji_var] = corr
                        optimal_lags.loc[coater_var, choji_var] = lag

    print("상세 시차 분석이 완료되었습니다.")
    max_corrs.replace(-2.0, np.nan, inplace=True) # 상관관계 계산 안된 곳은 NaN으로

    # Heatmap 1: Optimal Lags
    lag_heatmap_filename = 'excel_task3_optimal_lags_heatmap.png'
    plt.figure(figsize=(14, 12));
    ax1 = sns.heatmap(optimal_lags, annot=True, cmap='YlGnBu', fmt='.0f', linewidths=.5)
    if font_prop:
        ax1.set_title('변수별 최적 시차(Lag) 분석 (일)', fontproperties=font_prop, fontsize=16)
        for label in ax1.get_xticklabels(): label.set_fontproperties(font_prop)
        for label in ax1.get_yticklabels(): label.set_fontproperties(font_prop)
        for text in ax1.texts: text.set_font_properties(font_prop)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(lag_heatmap_filename); plt.close()
    print(f"'{lag_heatmap_filename}' 파일로 저장했습니다.")

    # Heatmap 2: Max Correlations
    corr_heatmap_filename = 'excel_task3_max_correlations_heatmap.png'
    plt.figure(figsize=(14, 12));
    ax2 = sns.heatmap(max_corrs, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5)
    if font_prop:
        ax2.set_title('최적 시차 적용 시 최대 상관계수', fontproperties=font_prop, fontsize=16)
        for label in ax2.get_xticklabels(): label.set_fontproperties(font_prop)
        for label in ax2.get_yticklabels(): label.set_fontproperties(font_prop)
        for text in ax2.texts: text.set_font_properties(font_prop)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(corr_heatmap_filename); plt.close()
    print(f"'{corr_heatmap_filename}' 파일로 저장했습니다.")

def main():
    font_prop = get_korean_font_prop()
    plt.rcParams['axes.unicode_minus'] = False

    print("엑셀 데이터를 전처리하고 있습니다...")
    choji_df, original_choji_cols, coater_df, original_coater_cols = preprocess_data('original_data_for_week1.xlsx')
    
    if choji_df.empty or coater_df.empty:
        print("데이터 처리 실패 또는 데이터 부족으로 분석을 중단합니다.")
        return

    print("데이터 전처리가 완료되었습니다.")

    print("\n[Task 1] 공정 내 변수들의 관계를 분석합니다...")
    coater_ignore_cols = [col for col in original_coater_cols if '평량' in col]
    plot_intra_process_correlation(choji_df, original_choji_cols, '초지 공정 내 변수들의 상관관계', 'excel_task1_choji_intra_correlation.png', font_prop)
    plot_intra_process_correlation(coater_df, original_coater_cols, '코타 공정 내 변수들의 상관관계', 'excel_task1_coater_intra_correlation.png', font_prop, cols_to_ignore=coater_ignore_cols)

    print("\n[Task 2] 코타 공정의 주요 인자가 품질에 미치는 영향을 분석합니다...")
    plot_coater_impact_analysis(coater_df, original_coater_cols, 'excel_task2_coater', font_prop)

    find_and_plot_detailed_influence(choji_df, original_choji_cols, coater_df, original_coater_cols, font_prop)
    
    print("\n모든 분석 및 시각화가 완료되었습니다!")

if __name__ == '__main__':
    main()

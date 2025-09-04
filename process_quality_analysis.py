ㅇ# -*- coding: utf-8 -*- 
# 이 코드는 공정 품질 데이터를 분석하고 시각화하는 프로그램입니다.
# 초지 공정과 코타 공정이라는 두 가지 주요 공정의 데이터를 분석하여,
# 각 공정 내에서 관리할 수 있는 요인들이 품질에 어떤 영향을 미치는지,
# 그리고 초지 공정의 변화가 코타 공정의 품질에 시간 지연을 두고 어떤 영향을 미치는지 알아봅니다.

# 필요한 도구들을 불러옵니다.
# pandas는 데이터를 표(데이터프레임) 형태로 다루는 데 사용됩니다.
# seaborn과 matplotlib.pyplot은 데이터를 그래프로 그려 시각화하는 데 사용됩니다.
# numpy는 숫자 계산에 필요한 도구입니다.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 데이터 불러오기 및 준비 ---

# 분석할 CSV 파일의 경로를 지정합니다.
file_path = '/workspaces/hansol25_week1/week1_data.csv'

try:
    # CSV 파일을 읽어옵니다.
    # encoding='cp949': 한글이 깨지지 않도록 한국어 인코딩 방식을 사용합니다.
    # header=1: 파일의 첫 번째 줄은 건너뛰고, 두 번째 줄을 표의 제목(컬럼명)으로 사용합니다.
    df = pd.read_csv(file_path, encoding='cp949', header=1)

    # 컬럼명(표의 제목)의 양 끝에 있는 공백을 제거합니다.
    # 이렇게 하면 컬럼명을 정확하게 찾을 수 있습니다.
    df.columns = df.columns.str.strip()

    # Matplotlib에서 한글 폰트가 깨지지 않도록 설정합니다.
    # 'NanumGothic' 폰트를 사용하도록 지정합니다. (리눅스 환경에 설치된 나눔고딕 폰트)
    plt.rcParams['font.family'] = 'NanumGothic'
    # 마이너스 부호가 깨지는 것을 방지합니다.
    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. 공정별 데이터 분리 ---

    # 전체 데이터에서 '작업장' 컬럼을 기준으로 'PM23 초지릴' 데이터만 따로 분리합니다.
    # .copy()를 사용하여 원본 데이터에 영향을 주지 않도록 복사본을 만듭니다.
    df_초지 = df[df['작업장'] == 'PM23 초지릴'].copy()
    # 'CM22 코타와인더' 데이터만 따로 분리합니다.
    df_코타 = df[df['작업장'] == 'CM22 코타와인더'].copy()

    # --- 3. 데이터 전처리 함수 정의 ---

    # 데이터를 분석하기 좋게 정리하는 함수를 만듭니다.
    # 이 함수는 숫자형 컬럼으로 변환하고, 일별 평균을 계산합니다.
    def process_df(dataframe):
        # 원본 데이터프레임을 복사하여 사용합니다.
        df_processed = dataframe.copy()

        # 숫자형으로 변환할 가능성이 있는 컬럼들의 목록입니다.
        # 사용자 요청에 따라 '속도', '선압', '선압.1'은 제외했습니다.
        numerical_cols_candidates = [
            'BP평량', '거칠음도_B', '거칠음도_T', '도공량_THERMAL', '도공량_UNDER',
            '동적발색도_EPSON', '동적발색_0.8msec', '동적발색_1.28msec', '두께',
            '백감도', '백색도', '불투명도', '색상_a', '색상_b', '색상_L',
            '인열강도_MD', '인장강도_MD',
            '정적발색_110℃', '정적발색_70℃', '지합', '투기도', '평량', '평량.1', '평활도_B', '평활도_T', '회분(배부)'
        ]

        # 현재 데이터프레임에 실제로 존재하는 컬럼들만 선택합니다.
        cols_to_process = [col for col in numerical_cols_candidates if col in df_processed.columns]
        
        # 선택된 컬럼들을 하나씩 살펴보면서 숫자형으로 변환합니다.
        for col in cols_to_process:
            # 만약 컬럼의 데이터 타입이 'object'(문자열 등)라면,
            if df_processed[col].dtype == 'object':
                # 숫자 안에 쉼표(,)가 있는 경우(예: "1,000") 쉼표를 제거합니다.
                df_processed[col] = df_processed[col].astype(str).str.replace(',', '', regex=False)
            # 컬럼의 데이터를 숫자로 변환합니다.
            # errors='coerce': 숫자로 변환할 수 없는 값은 NaN(Not a Number, 비어있음)으로 만듭니다.
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # '일자' 컬럼을 날짜 시간 형식으로 변환합니다.
        df_processed['일자'] = pd.to_datetime(df_processed['일자'])
        # '일자' 컬럼을 데이터프레임의 인덱스(기준)로 설정합니다.
        df_processed = df_processed.set_index('일자')
        
        # 숫자형 컬럼들만 선택하여 일별 평균을 계산합니다.
        # .resample('D').mean(): 날짜를 기준으로 하루 단위로 묶어서 평균을 냅니다.
        df_daily_avg = df_processed[cols_to_process].resample('D').mean()
        return df_daily_avg

    # 초지 공정과 코타 공정 데이터에 위에서 만든 함수를 적용하여 일별 평균 데이터를 만듭니다.
    df_초지_daily = process_df(df_초지)
    df_코타_daily = process_df(df_코타)

    # --- 4. 공정 내 상관관계 분석 및 시각화 ---

    print("\n--- 4. 공정 내 상관관계 분석 ---")

    # 분석할 공정 목록과 각 공정의 일별 평균 데이터입니다.
    processes = {
        "초지": df_초지_daily,
        "코타": df_코타_daily
    }

    # 제어 가능한 요인 목록입니다.
    controllable_factors = ['BLADE_시간', '도공량_THERMAL', '도공량_UNDER']
    # 주요 품질 요인 목록입니다.
    key_quality_factors = ['백색도', '평활도_B', '거칠음도_B', '동적발색도_EPSON']

    # 각 공정별로 상관관계를 분석하고 히트맵을 그립니다.
    for proc_name, proc_df in processes.items():
        print(f"\n--- {proc_name} 공정 내 관리 인자와 품질 요인 상관관계 ---")
        
        # 현재 공정 데이터에 존재하는 제어 가능한 요인들만 선택합니다.
        current_controllable_factors = [f for f in controllable_factors if f in proc_df.columns]
        # 현재 공정 데이터에 존재하는 주요 품질 요인들만 선택합니다.
        current_key_quality_factors = [q for q in key_quality_factors if q in proc_df.columns]

        # 만약 제어 가능한 요인과 품질 요인이 모두 존재한다면 상관관계를 계산합니다.
        if current_controllable_factors and current_key_quality_factors:
            # 제어 가능한 요인과 품질 요인 컬럼만 선택합니다.
            # .dropna(): NaN(비어있는 값)이 있는 행은 제거합니다.
            # .corr(): 상관관계를 계산합니다.
            correlation_subset = proc_df[current_controllable_factors + current_key_quality_factors].dropna().corr()
            
            # 제어 가능한 요인과 품질 요인 간의 상관관계 부분만 추출합니다.
            # .loc[]를 사용하여 특정 행과 열을 선택합니다.
            # 상관관계 행렬에서 제어 가능한 요인(행)과 품질 요인(열)이 만나는 부분을 가져옵니다.
            specific_correlations = correlation_subset.loc[current_controllable_factors, current_key_quality_factors]
            
            print(f"\n{proc_name} 공정 관리 인자 vs 품질 요인 상관관계:")
            print(specific_correlations)

            # 상관관계 히트맵을 그립니다.
            plt.figure(figsize=(10, 8)) # 그래프의 크기를 설정합니다.
            # sns.heatmap(): 히트맵을 그리는 함수입니다.
            # annot=True: 각 칸에 상관계수 값을 표시합니다.
            # cmap='coolwarm': 색상 테마를 지정합니다.
            # fmt=".2f": 소수점 둘째 자리까지 표시합니다.
            # linewidths=.5: 칸 사이에 선을 그립니다.
            sns.heatmap(specific_correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title(f'{proc_name} 공정 관리 인자 vs 품질 요인 상관관계 히트맵', fontsize=16) # 그래프 제목
            plt.xticks(rotation=45, ha='right') # x축 라벨을 45도 회전
            plt.yticks(rotation=0) # y축 라벨은 회전하지 않음
            plt.tight_layout() # 그래프 요소들이 겹치지 않도록 자동으로 조정
            # 그래프를 이미지 파일로 저장합니다.
            plt.savefig(f'/workspaces/hansol25_week1/{proc_name}_intra_process_correlation_heatmap.png')
            plt.close() # 현재 그래프 창을 닫습니다.
            print(f"'{proc_name}_intra_process_correlation_heatmap.png' 파일이 저장되었습니다.")
        else:
            print(f"{proc_name} 공정에는 분석할 관리 인자 또는 품질 요인이 충분하지 않습니다.")

    # --- 5. 초지-코타 공정 간 시간 지연 상관관계 분석 및 시각화 ---

    print("\n--- 5. 초지-코타 공정 간 시간 지연 상관관계 분석 ---")

    # 두 공정의 일별 평균 데이터를 병합합니다.
    # on='일자': '일자' 컬럼을 기준으로 병합합니다.
    # how='outer': 양쪽 데이터에 있는 모든 날짜를 포함합니다.
    # suffixes: 컬럼명이 겹칠 경우 뒤에 붙일 접미사입니다.
    df_merged_daily = pd.merge(df_초지_daily, df_코타_daily, on='일자', how='outer', suffixes=('_초지', '_코타'))
    
    # 초지 공정의 품질/관리 지표와 코타 공정의 품질 지표를 정의합니다.
    # 이 지표들을 사용하여 시간 지연 상관관계를 분석할 것입니다.
    초지_metrics_for_lag_analysis = {
        '백색도': '백색도_초지',
        '거칠음도_B': '거칠음도_B_초지',
        'BLADE_시간': 'BLADE_시간_초지', # 제어 요인 포함
        '도공량_THERMAL': '도공량_THERMAL_초지', # 제어 요인 포함
        '도공량_UNDER': '도공량_UNDER_초지' # 제어 요인 포함
    }
    코타_metrics_for_lag_analysis = {
        '백색도': '백색도_코타',
        '평활도_B': '평활도_B_코타',
        '동적발색도_EPSON': '동적발색도_EPSON_코타'
    }

    # 병합된 데이터프레임에 실제로 존재하는 컬럼들만 선택합니다.
    초지_metrics_for_lag_analysis = {k: v for k, v in 초지_metrics_for_lag_analysis.items() if v in df_merged_daily.columns}
    코타_metrics_for_lag_analysis = {k: v for k, v in 코타_metrics_for_lag_analysis.items() if v in df_merged_daily.columns}

    # 시간 지연(Lag) 범위를 설정합니다. 0일부터 7일까지 테스트합니다.
    lags = range(0, 8)
    # 분석 결과를 저장할 목록입니다.
    lag_analysis_results = []

    # 초지 공정의 각 지표와 코타 공정의 각 지표를 조합하여 시간 지연 상관관계를 계산합니다.
    for 초지_name, 초지_col in 초지_metrics_for_lag_analysis.items():
        for 코타_name, 코타_col in 코타_metrics_for_lag_analysis.items():
            correlations = [] # 각 지연별 상관계수를 저장할 목록
            for lag in lags:
                # 초지 데이터를 'lag' 일만큼 뒤로 미룹니다.
                # 예를 들어, lag이 1이면 초지 데이터의 N일 값이 코타 데이터의 N+1일 값과 비교됩니다.
                # 이는 초지 공정의 N일 결과가 N+lag일 코타 공정 결과에 영향을 미친다는 가정을 반영합니다.
                shifted_초지_col = df_merged_daily[초지_col].shift(lag)
                
                # 지연된 초지 데이터와 코타 데이터 간의 상관관계를 계산합니다.
                # .corr(): 상관계수를 계산하며, NaN(비어있는 값)은 자동으로 무시합니다.
                correlation = shifted_초지_col.corr(df_merged_daily[코타_col])
                correlations.append(correlation) # 계산된 상관계수를 목록에 추가
            
            # 분석 결과를 저장합니다.
            lag_analysis_results.append({
                '초지_Metric': 초지_name,
                '코타_Metric': 코타_name,
                'Correlations': correlations
            })
            
            # 계산된 시간 지연 상관관계를 출력합니다.
            print(f"\n{초지_name}_초지 vs {코타_name}_코타 (시간 지연 상관관계):")
            for i, corr in enumerate(correlations):
                print(f"  Lag {i} days: {corr:.2f}")

    # --- 6. 시각화: 시계열 변화 및 교차 상관관계 플롯 ---

    print("\n--- 6. 시각화 결과 ---")

    # 시계열 변화 플롯을 그립니다.
    # 각 초지-코타 지표 쌍에 대해 시간 흐름에 따른 변화를 보여줍니다.
    for 초지_name, 초지_col in 초지_metrics_for_lag_analysis.items():
        for 코타_name, 코타_col in 코타_metrics_for_lag_analysis.items():
            plt.figure(figsize=(14, 7)) # 그래프 크기 설정
            # 초지 공정 지표를 선 그래프로 그립니다.
            plt.plot(df_merged_daily.index, df_merged_daily[초지_col], label=f'초지 {초지_name}', marker='o', linestyle='-', markersize=4)
            # 코타 공정 지표를 다른 스타일의 선 그래프로 그립니다.
            plt.plot(df_merged_daily.index, df_merged_daily[코타_col], label=f'코타 {코타_name}', marker='x', linestyle='--', markersize=4)
            plt.title(f'일별 {초지_name} (초지) 및 {코타_name} (코타) 변화 추이') # 그래프 제목
            plt.xlabel('날짜') # x축 라벨
            plt.ylabel('값') # y축 라벨
            plt.legend() # 범례 표시
            plt.grid(True) # 그리드 표시
            plt.tight_layout() # 레이아웃 자동 조정
            # 그래프를 이미지 파일로 저장합니다.
            plt.savefig(f'/workspaces/hansol25_week1/time_series_{초지_name}_vs_{코타_name}.png')
            plt.close() # 현재 그래프 창을 닫습니다.
            print(f"'{초지_name}_vs_{코타_name}_time_series.png' 파일이 저장되었습니다.")

    # 교차 상관관계 플롯을 그립니다.
    # 각 지연(Lag) 값에 따른 상관계수의 변화를 보여줍니다.
    for res in lag_analysis_results:
        초지_name = res['초지_Metric']
        코타_name = res['코타_Metric']
        correlations = res['Correlations']

        plt.figure(figsize=(10, 6)) # 그래프 크기 설정
        # 지연(Lag)과 상관계수를 선 그래프로 그립니다.
        plt.plot(lags, correlations, marker='o', linestyle='-')
        plt.title(f'교차 상관관계: {초지_name} (초지) vs {코타_name} (코타)') # 그래프 제목
        plt.xlabel('시간 지연 (일)') # x축 라벨
        plt.ylabel('상관관계 계수') # y축 라벨
        plt.xticks(lags) # x축 눈금을 지연 값으로 설정
        plt.grid(True) # 그리드 표시
        plt.tight_layout() # 레이아웃 자동 조정
        # 그래프를 이미지 파일로 저장합니다.
        plt.savefig(f'/workspaces/hansol25_week1/cross_correlation_{초지_name}_vs_{코타_name}.png')
        plt.close() # 그래프 창 닫기
        print(f"'{초지_name}_vs_{코타_name}_cross_correlation.png' 파일이 저장되었습니다.")

    # --- 7. 분석 결과 해석 및 제안 ---

    print("\n--- 7. 분석 결과 해석 및 제안 ---")
    print("이 분석은 초지 공정의 특정 요인들이 코타 공정의 품질에 어떤 시간 지연을 두고 영향을 미치는지 파악하는 데 도움을 줍니다.")
    print("교차 상관관계 플롯에서 상관계수가 가장 높은 지연(Lag) 값을 찾아보세요. 이 지연 값이 초지 공정의 변화가 코타 공정 품질에 영향을 미치는 데 걸리는 대략적인 시간을 나타낼 수 있습니다.")
    print("\n**주요 해석 가이드:**")
    print("1. **최적 지연(Lag) 찾기:** 각 '교차 상관관계' 그래프에서 상관계수(y축)가 가장 높거나 낮은(절대값이 큰) 지점의 시간 지연(x축)을 확인하세요. 이 지연이 한 공정의 변화가 다른 공정에 영향을 미치는 데 걸리는 시간일 수 있습니다.")
    print("2. **영향의 방향:**")
    print("   - **양의 상관관계 (양수 값):** 한 지표가 증가하면 다른 지표도 증가하고, 한 지표가 감소하면 다른 지표도 감소하는 경향이 있습니다. (예: 초지 백색도가 높아지면 코타 백색도도 높아짐)")
    print("   - **음의 상관관계 (음수 값):** 한 지표가 증가하면 다른 지표는 감소하고, 반대로 한 지표가 감소하면 다른 지표는 증가하는 경향이 있습니다. (예: 초지 거칠음도가 높아지면 코타 평활도는 낮아짐)")
    print("3. **상관관계의 강도:**")
    print("   - **0.7 이상 (또는 -0.7 이하):** 매우 강한 상관관계")
    print("   - **0.5 ~ 0.7 (또는 -0.5 ~ -0.7):** 강한 상관관계")
    print("   - **0.3 ~ 0.5 (또는 -0.3 ~ -0.5):** 보통 상관관계")
    print("   - **0.1 ~ 0.3 (또는 -0.1 ~ -0.3):** 약한 상관관계")
    print("   - **0.1 미만 (또는 -0.1 초과):** 거의 상관관계 없음")
    print("\n**분석의 한계점 및 추가 고려사항:**")
    print("1. **데이터 정합성:** 이 분석은 일별 평균 데이터를 기반으로 하므로, 실제 생산 배치 단위의 미세한 시간 지연이나 복잡한 상호작용은 반영하지 못할 수 있습니다. 더 정확한 분석을 위해서는 각 생산 배치에 대한 고유 식별자(Batch ID)와 공정 간의 정확한 시간 흐름 정보가 필요합니다.")
    print("2. **인과관계 아님:** 상관관계가 높다고 해서 반드시 한 요인이 다른 요인의 원인이라는 의미는 아닙니다. 숨겨진 다른 요인이 두 지표 모두에 영향을 미칠 수도 있습니다. 실제 공정 개선을 위해서는 전문가의 지식과 추가적인 실험이 필요합니다.")
    print("3. **데이터 부족:** 특정 날짜에 데이터가 없거나, 특정 지표의 데이터가 부족한 경우 상관관계 계산에 영향을 미칠 수 있습니다.")
    print("이 분석 결과는 공정 개선을 위한 아이디어를 얻는 데 활용될 수 있습니다. 공정 전문가의 경험과 결합하여 최적의 개선 방안을 찾아보세요.")

# 파일을 찾을 수 없을 때 발생하는 오류를 처리합니다.
except FileNotFoundError:
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
# 그 외 예상치 못한 오류가 발생했을 때 처리합니다.
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")

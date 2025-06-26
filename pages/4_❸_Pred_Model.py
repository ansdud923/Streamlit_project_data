import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time  
import folium
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go


# [파스텔톤 Hex Codes]
# 파스텔 블루: #ADD8E6
# 파스텔 그린: #77DD77
# 파스텔 퍼플: #B19CD9
# 파스텔 옐로우: #FFFACD
# 파스텔 피치: #FFDAB9
# 파스텔 민트: #BDFCC9
# 파스텔 라벤더: #E6E6FA
# 파스텔 노란색: #FFF44F
# 파스텔 그린: #B2FBA5

# 메인 페이지 너비 넓게 (가장 처음에 설정해야 함)
st.set_page_config(layout="wide") 

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  # 대기 시간 시뮬레이션
st.success("Data Loaded!")

# 한글 및 마이너스 깨짐
plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# 클라우드 배포 시 한글, 마이너스 깨짐 방지


# CSV 파일 경로 설정
CSV_FILE_PATH = 'https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/'


member_df = pd.read_csv(CSV_FILE_PATH + 'members_data.csv')
# member_df = pd.read_csv('https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/members_data.csv')

# Streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

#온/오프라인 데이터 로드
@st.cache_data
def on_load_data():
    df_on = pd.read_csv(CSV_FILE_PATH + 'recycling_online.csv', encoding="UTF8").fillna(0)
    df_on.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_on.fillna(0, inplace=True)
    return df_on

@st.cache_data
def off_load_data():
    df_off = pd.read_csv(CSV_FILE_PATH + 'recycling_off.csv', encoding="UTF8")
    df_off.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_off.dropna(subset=["날짜"], inplace=True)
    return df_off

df_on = on_load_data()
df_off = off_load_data()

print_df = member_df.rename(columns={
     "age": "나이",
     "gender": "성별",
     "marriage": "혼인여부",
     "city": "도시",
     "channel": "가입경로",
     "before_ev": "참여_전",
     "part_ev": "참여이벤트",
     "after_ev": "참여_후"
})

# 데이터값 변경
print_df['성별'] = print_df['성별'].map({0:'남자', 1:'여자'})
print_df['혼인여부'] = print_df['혼인여부'].map({0:'미혼', 1:'기혼'})
print_df['도시'] = print_df['도시'].map({0:'부산', 1:'대구', 2:'인천', 3:'대전', 4:'울산', 5:'광주', 6:'서울', 
    7:'경기', 8:'강원', 9:'충북', 10:'충남', 11:'전북', 12:'전남', 13:'경북', 14:'경남', 15:'세종', 16:'제주'})
print_df['가입경로'] = print_df['가입경로'].map({0:"직접 유입", 1:"키워드 검색", 2:"블로그", 3:"카페", 4:"이메일", 
        5:"카카오톡", 6:"메타", 7:"인스타그램", 8:"유튜브", 9:"배너 광고", 10:"트위터 X", 11:"기타 SNS"})
print_df['참여_전'] = print_df['참여_전'].map({0:'가입', 1:'미가입'})
print_df['참여이벤트'] = print_df['참여이벤트'].map({0:"워크숍 개최", 1:"재활용 품목 수집 이벤트", 2:"재활용 아트 전시",
          3:"게임 및 퀴즈", 4:"커뮤니티 청소 활동", 5:"업사이클링 마켓", 6:"홍보 부스 운영"})
print_df['참여_후'] = print_df['참여_후'].map({0:'가입', 1:'미가입'})

# 특성 공학 함수 추가
def create_features(data):
    """특성 공학을 통해 새로운 변수 생성"""
    data_copy = data.copy()
    
    # 연령대 그룹 생성
    data_copy['age_group'] = pd.cut(data_copy['age'], 
                                   bins=[0, 30, 40, 50, 100], 
                                   labels=['20대', '30대', '40대', '50대이상'])
    
    # 성별-혼인상태 조합 변수
    data_copy['gender_marriage'] = data_copy['gender'].astype(str) + '_' + data_copy['marriage'].astype(str)
    
    # 도시 규모별 그룹 (대도시, 중소도시 등)
    metro_cities = [6, 7]  # 서울, 경기
    major_cities = [0, 1, 2, 3, 4, 5]  # 부산, 대구, 인천, 대전, 울산, 광주
    data_copy['city_type'] = data_copy['city'].apply(
        lambda x: 'metro' if x in metro_cities else 'major' if x in major_cities else 'other'
    )
    
    return data_copy

data = create_features(member_df[['age', 'city', 'gender', 'marriage', 'after_ev']])

tab1, tab2, tab3, tab4, tab5 = st.tabs(['서비스가입 예측', '추천 캠페인', '추천 채널', '전환율 예측', '방문자수 예측'])

with tab1: # 서비스 가입 예측 모델
    with st.expander('회원 데이터'):
        st.dataframe(print_df, use_container_width=True)
    col1, col2, col3 = st.columns([4, 3, 3])
    with col1:
        st.write("서비스가입 예측 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_1 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45)
        )
        st.write(f"**선택 연령대: :red[{ages_1}]세**")

    with col2:
        gender_1 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=0
        )
    
    with col3:
        marriage_1 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=0
        )
    
    # 예측 모델 학습 및 평가 함수
    @st.cache_data
    def train_model(data):
        numeric_features = ['age']
        categorical_features = ['gender', 'marriage', 'city', 'age_group', 'gender_marriage', 'city_type']

        # ColumnTransformer 설정
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features), # 수치형 - 표준화 
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features) # 범주형 - 원핫인코딩
            ]
        )

        # 여러 모델 정의
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )

        # 앙상블 모델 생성
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', Pipeline([('preprocessor', preprocessor), ('classifier', rf_model)])),
                ('gb', Pipeline([('preprocessor', preprocessor), ('classifier', gb_model)])),
                ('lr', Pipeline([('preprocessor', preprocessor), ('classifier', lr_model)]))
            ],
            voting='soft'
        )

        # 데이터 분할 (계층화 샘플링)
        X = data.drop(columns=['after_ev'])
        y = data['after_ev']
        
        # 계층화 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 교차검증을 통한 모델 평가
        cv_scores = cross_val_score(ensemble_model, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='f1')
        
        # 모델 학습
        ensemble_model.fit(X_train, y_train)

        return ensemble_model, X_test, y_test, cv_scores

    # 성능 평가 및 지표 출력 함수
    def evaluate_model(model, X_test, y_test, cv_scores):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 교차검증 결과 포함
        st.write(f"**교차검증 F1 점수: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})**")
        st.write(f"**테스트 성능 - 정확도: {accuracy * 100:.1f}%, 정밀도: {precision * 100:.1f}%, 재현율: {recall * 100:.1f}%**")
        st.write(f"**F1-Score: {f1 * 100:.1f}%**")

        # 상세 분류 리포트
        with st.expander("상세 분류 리포트"):
            st.text(classification_report(y_test, y_pred, target_names=['가입', '미가입']))

        return y_pred, y_pred_proba

    # 시각화 함수 (혼동 행렬 및 ROC 곡선)
    def plot_metrics(y_test, y_pred, y_pred_proba):
        cm = confusion_matrix(y_test, y_pred)
        
        y_scores = y_pred_proba  # 긍정 클래스 확률
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # 첫 번째 열에 혼동 행렬 시각화
        col1, col2 = st.columns(2)

        with col1:
            # 혼동 행렬 시각화
            cm_df = pd.DataFrame(cm, index=['미가입', '가입'], columns=['미가입', '가입'])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='GnBu', 
                            title='혼동 행렬 (Confusion Matrix)')
            fig.update_xaxes(title='예측 레이블')
            fig.update_yaxes(title='실제 레이블')
            fig.update_layout(width=600, height=600)
            st.plotly_chart(fig)
            
            # 혼동 행렬 해석 추가
            tn, fp, fn, tp = cm.ravel()
            st.write("**혼동 행렬 해석:**")
            st.write(f"- 참 음성 (TN): {tn}개 - 미가입을 미가입으로 정확히 예측")
            st.write(f"- 거짓 양성 (FP): {fp}개 - 미가입을 가입으로 잘못 예측")  
            st.write(f"- 거짓 음성 (FN): {fn}개 - 가입을 미가입으로 잘못 예측")
            st.write(f"- 참 양성 (TP): {tp}개 - 가입을 가입으로 정확히 예측")
    
        with col2:
            # ROC 곡선 시각화
            fig_roc = go.Figure()
            
            # ROC 곡선 추가
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, 
                mode='lines', 
                name=f'ROC curve (AUC = {roc_auc:.3f})', 
                line=dict(width=2, color='blue')
            ))
            
            # 랜덤 분류기 (대각선) 추가
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], 
                mode='lines', 
                name='Random Classifier (AUC = 0.5)', 
                line=dict(width=2, dash='dash', color='red')
            ))
            
            # 레이아웃 설정
            fig_roc.update_layout(
                title='ROC 곡선 (Receiver Operating Characteristic)',
                xaxis_title='False Positive Rate (거짓 양성률)',
                yaxis_title='True Positive Rate (참 양성률)',
                showlegend=True,
                width=600,
                height=600,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            # Streamlit에서 ROC 곡선 그래프 표시
            st.plotly_chart(fig_roc)
            
            # ROC AUC 해석 추가
            st.write("**ROC AUC 해석:**")
            if roc_auc >= 0.9:
                st.write(f"🟢 AUC = {roc_auc:.3f} - 매우 우수한 성능")
            elif roc_auc >= 0.8:
                st.write(f"🔵 AUC = {roc_auc:.3f} - 우수한 성능")
            elif roc_auc >= 0.7:
                st.write(f"🟡 AUC = {roc_auc:.3f} - 양호한 성능")
            elif roc_auc >= 0.6:
                st.write(f"🟠 AUC = {roc_auc:.3f} - 보통 성능")
            else:
                st.write(f"🔴 AUC = {roc_auc:.3f} - 개선 필요")

    # 예측 결과 출력 함수
    def pre_result(model, new_data):
        prediction = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)

        result_text = '가입' if prediction[0] == 0 else '미가입'
        confidence = prediction_proba[0][prediction[0]] * 100
        
        st.write(f"**모델 예측 결과: :rainbow[{result_text}] (신뢰도: {confidence:.1f}%)**")

    # 버튼 클릭에 따른 동작
    if st.button("예측하기"):
        # 입력된 값을 새로운 데이터 형식으로 변환
        new_data = pd.DataFrame({
            'age': [(ages_1[0] + ages_1[1]) / 2],
            'gender': [1 if gender_1 == '여자' else 0],
            'marriage': [1 if marriage_1 == '기혼' else 0],
            'city': [6]  # 기본값으로 서울 설정
        })

        # 특성 공학 적용
        new_data = create_features(new_data)

        # 기존 데이터로 모델 학습
        ensemble_model, X_test, y_test, cv_scores = train_model(data)

        # 예측 수행
        pre_result(ensemble_model, new_data)

        # 성능 평가 및 지표 출력
        y_pred, y_pred_proba = evaluate_model(ensemble_model, X_test, y_test, cv_scores)

        # 시각화
        plot_metrics(y_test, y_pred, y_pred_proba)


# 캠페인 추천 모델
data_2 = create_features(member_df[['age', 'city', 'gender', 'marriage', 'before_ev', 'part_ev', 'after_ev']])

# 참여 이벤트 매핑
event_mapping = {
    0: "워크숍 개최",
    1: "재활용 품목 수집 이벤트",
    2: "재활용 아트 전시",
    3: "게임 및 퀴즈",
    4: "커뮤니티 청소 활동",
    5: "업사이클링 마켓",
    6: "홍보 부스 운영"
}

city_mapping = {
    0: '부산',
    1: '대구', 
    2: '인천', 
    3: '대전', 
    4: '울산', 
    5: '광주', 
    6: '서울', 
    7: '경기', 
    8: '강원', 
    9: '충북', 
    10: '충남', 
    11: '전북', 
    12: '전남', 
    13: '경북', 
    14: '경남', 
    15: '세종', 
    16: '제주'
}

with tab2: # 캠페인 추천 모델
    with st.expander('회원 데이터'):
        st.dataframe(print_df, use_container_width=True)
    col1, col2, col3 = st.columns([4, 3, 3])
    with col1:
        st.write("캠페인 추천 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_2 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_2'
        )
        st.write(f"**선택 연령대: :red[{ages_2}]세**")
        
    with col2:
        gender_2 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=0,
            key='radio2_1'
        )
    
    with col3:
        marriage_2 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=0,
            key='radio2_2'
        )

    # 추천 모델 함수
    @st.cache_data
    def calculate_enrollment_increase_rate(data):
        #캠페인 별 가입 증가율 계산
        increase_rates = {}
        campaign_stats = {}
        
        # 조건별 캠페인 그룹화 및 계산
        campaign_groups = data.groupby('part_ev')
        
        for campaign, group in campaign_groups:
            # 캠페인전과 후의 가입자 수 계산
            pre_signups = (group['before_ev'] == 0).sum()  # 캠페인 전 가입자 수 (0의 수)
            post_signups = (group['after_ev'] == 0).sum()  # 캠페인 후 가입자 수 (0의 수)
            total_participants = len(group)
            
            # 전환율 계산
            if total_participants > 0:
                conversion_rate = post_signups / total_participants
                # 베이즈 추정으로 신뢰구간 계산
                alpha = post_signups + 1
                beta = total_participants - post_signups + 1
                mean_rate = alpha / (alpha + beta)
                
                increase_rates[campaign] = mean_rate
                campaign_stats[campaign] = {
                    'conversion_rate': conversion_rate,
                    'participants': total_participants,
                    'signups': post_signups
                }
            else:
                increase_rates[campaign] = 0
                campaign_stats[campaign] = {
                    'conversion_rate': 0,
                    'participants': 0,
                    'signups': 0
                }

        return increase_rates, campaign_stats

    def recommend_campaign(data, age_range, gender, marriage):
    # 조건에 따라 데이터 필터링
        filtered_data = data[
            (data['age'].between(age_range[0], age_range[1])) &
            (data['gender'] == (1 if gender == '여자' else 0)) &
            (data['marriage'] == (1 if marriage == '기혼' else 0))
        ]

        if filtered_data.empty:
            return "해당 조건에 맞는 데이터가 없습니다."
        
        # 가입 증가율 계산
        increase_rates, campaign_stats = calculate_enrollment_increase_rate(filtered_data)

        # 가장 높은 가입 증가율을 가진 캠페인 추천
        best_campaign = max(increase_rates, key=increase_rates.get)
        
        return best_campaign, increase_rates, campaign_stats

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("캠페인 추천 받기"):
        best_campaign, increase_rates, campaign_stats = recommend_campaign(data_2, ages_2, gender_2, marriage_2)
            
        if isinstance(best_campaign, str):
            st.write(best_campaign)
        else:
            st.write(f"**추천 캠페인: :violet[{event_mapping[best_campaign]}] 👈 이 캠페인이 가장 가입을 유도할 수 있습니다!**")

            # 상세 통계 정보 출력
            best_stats = campaign_stats[best_campaign]
            st.write(f"**추천 근거: 참여자 {best_stats['participants']}명 중 {best_stats['signups']}명 가입 (전환율: {best_stats['conversion_rate']:.1%})**")
            
            # 가입 증가율 결과 출력
            with st.expander("**각 캠페인별 가입 증가율 보기**"):
                for campaign, rate in increase_rates.items():
                    stats = campaign_stats[campaign]
                    st.write(f"**{event_mapping[campaign]}**: 전환율 {rate:.1%} (참여자: {stats['participants']}명, 가입: {stats['signups']}명)")
            
            # 가입 증가율 결과 출력 및 가로 막대그래프 표시
            campaigns, rates = zip(*increase_rates.items())
            campaigns = [event_mapping[campaign] for campaign in campaigns]  # 매핑된 캠페인 이름
            
            # 파스텔 톤 색상 리스트 생성
            pastel_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#77DD77', '#B19CD9', '#FFDAB9']

            # 가로 막대그래프 시각화
            fig_bar = go.Figure()

            # 가로 막대 추가
            fig_bar.add_trace(go.Bar(
                y=campaigns,  # 캠페인 이름
                x=rates,      # 가입 증가율
                orientation='h',  # 가로 막대그래프
                marker=dict(color=pastel_colors),  # 색상 설정
            ))

            # 0 선 추가
            fig_bar.add_shape(
                type='line',
                x0=0,
                y0=-0.5,
                x1=0,
                y1=len(campaigns) - 0.5,
                line=dict(color='gray', width=0.8),
            )

            # 레이아웃 설정
            fig_bar.update_layout(
                title='캠페인별 가입 증가율',
                xaxis_title='가입 증가율',
                height=600
            )

            # X축 설정
            fig_bar.update_xaxes(
                range=[min(min(rates), 0), max(max(rates), 0)],  # X축 범위 설정
                showgrid=True
            )

            # Y축 설정
            fig_bar.update_yaxes(
                title='캠페인',
                showgrid=False
            )

            # Streamlit에서 가로 막대그래프 표시
            st.plotly_chart(fig_bar)

# 마케팅 채널 추천 
data_3 = member_df[['age', 'gender', 'marriage', 'channel', 'before_ev']]

# 가입 시 유입경로 매핑
register_channel = {
    0:"직접 유입",
    1:"키워드 검색",
    2:"블로그",
    3:"카페",
    4:"이메일",
    5:"카카오톡",
    6:"메타",
    7:"인스타그램",
    8:"유튜브", 
    9:"배너 광고", 
    10:"트위터 X", 
    11:"기타 SNS"
}

with tab3: # 마케팅 채널 추천 모델
    with st.expander('회원 데이터'):
        st.dataframe(print_df, use_container_width=True)
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.write("마케팅 채널 추천 모델입니다. 아래의 조건을 선택해 주세요")
        ages_3 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_3'
        )
        st.write(f"**선택 연령대: :red[{ages_3}]세**")
    
    with col2:
        gender_3 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=0,
            key='radio3_1'
        )
    
    with col3:
        marriage_3 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=0,
            key='radio3_2'
        )

    # 추천 모델 함수
    @st.cache_data
    def calculate_channel_conversion_rate(data):
        # 마케팅 채널별 가입률 계산
        channel_stats = data.groupby('channel').agg(
        total_members=('before_ev', 'count'),   # 전체 유입자 수
        total_signups=('before_ev', lambda x: (x == 0).sum())  # 가입자 수 (before_ev가 0인 경우)
        )
        
        # 가입률 계산: 가입자의 수 / 전체 유입자의 수
        channel_stats['conversion_rate'] = channel_stats['total_signups'] / channel_stats['total_members']
        channel_stats.reset_index(inplace=True)
        return channel_stats[['channel', 'conversion_rate']]

    def recommend_channel(data, age_range, gender, marriage):
        #조건에 맞는 가장 추천 마케팅 채널 3개를 반환
        filtered_data = data[
            (data['age'].between(age_range[0], age_range[1])) &
            (data['gender'] == (1 if gender == '여자' else 0)) &
            (data['marriage'] == (1 if marriage == '기혼' else 0))
        ]

        channel_rates = calculate_channel_conversion_rate(filtered_data)
        
        # "직접 유입" 채널 제외
        channel_rates = channel_rates[channel_rates['channel'] != 0]
        
        top_channels = channel_rates.nlargest(3, 'conversion_rate')
        
        return top_channels

    def display_channel_rates(channel_rates):
        #마케팅 채널 가입률 수치 표시
        with st.expander("**각 마케팅 채널별 가입률 보기**"):
            for _, row in channel_rates.iterrows():
                channel_name = register_channel[row['channel']]
                st.write(f"{channel_name}: {row['conversion_rate']:.2%}")

    def plot_channel_rates(channel_rates):
        #마케팅 채널 가입률 시각화 (막대 그래프)
        fig_bar = go.Figure()

        # 파스텔 톤 색상 리스트 생성
        pastel_colors = ['#FFDAB9', '#BDFCC9', '#E6E6FA']

        fig_bar.add_trace(go.Bar(
            y=channel_rates['channel'].apply(lambda x: register_channel[x]),
            x = channel_rates['conversion_rate'],
            orientation='h',
            marker=dict(color=pastel_colors),
        ))

        # 선추가
        fig_bar.add_shape(
            type='line',
            x0=0,
            y0=-0.5,
            x1=0,
            y1=len(channel_rates) - 0.5,  # Y축 개수
            line=dict(color='gray', width=0.8),
        )

        # 레이아웃 설정
        fig_bar.update_layout(
            title='마케팅 채널별 가입률',
            xaxis_title='가입률',
            height=600
        )

        # X축 설정
        fig_bar.update_xaxes(
            range=[min(min(channel_rates['conversion_rate']), 0), max(max(channel_rates['conversion_rate']), 0)],
            showgrid=True
        )

        # y축 설정
        fig_bar.update_yaxes(
            title='마케팅 채널',
            showgrid=False)
        
        # 표시
        st.plotly_chart(fig_bar)

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 마케팅 채널 추천받기"):
        # 추천 모델 훈련
        top_channels = recommend_channel(data_3, ages_3, gender_3, marriage_3)

        if not top_channels.empty:
            st.write(f"**추천 마케팅 채널:** :violet[{', '.join(top_channels['channel'].apply(lambda x: register_channel[x]))}] 👈 이 채널들이 가장 효과적입니다!")
            display_channel_rates(top_channels)
            plot_channel_rates(top_channels)
        else:
            st.write("해당 조건에 맞는 마케팅 채널이 없습니다.")

with tab4:  #전환율 예측
    # 데이터 로드
    with st.expander('온라인 데이터'):
        st.dataframe(df_on, use_container_width=True)
    select_all_device = st.checkbox("디바이스 전체 선택")
    device_options = df_on["디바이스"].unique().tolist()
    select_all_path = st.checkbox("유입경로 전체 선택")
    path_options = df_on["유입경로"].unique().tolist()

    if select_all_device:
        select_device = st.multiselect("디바이스", device_options, default = device_options)        
    else:
        select_device = st.multiselect("디바이스", device_options)

    if select_all_path:
        select_path = st.multiselect("유입경로", path_options, default = path_options)
    else:
        select_path = st.multiselect("유입경로", path_options)
    time_input = st.slider("체류 시간(분)", min_value = 0, max_value = 100, value = 0, step = 5)
        
    #온라인 데이터 복사
    df_ml_on = df_on.copy()

    # 결측값 처리 (학습 전에 미리 처리)
    df_ml_on["전환율(가입)"] = df_ml_on["전환율(가입)"].fillna(df_ml_on["전환율(가입)"].median())
    df_ml_on["체류시간(min)"] = df_ml_on["체류시간(min)"].fillna(df_ml_on["체류시간(min)"].median())

    # 원-핫 인코딩
    df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"], drop_first=False)         

    #체류시간 및 원-핫 인코딩된 디바이스, 유입경로 및 타겟 변수 설정
    features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
    target = "전환율(가입)"

    if st.button("온라인 전환율 예측"):
        # 데이터 유효성 검사
        if df_ml_on[target].isnull().sum() > 0:
            st.warning("데이터에 결측값이 있습니다. 전처리를 확인해주세요.")
            
        #입력(X), 출력(y) 데이터 정의
        X = df_ml_on[features]
        y = df_ml_on[target]

        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        #학습 데이터와 테스트 데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42, shuffle=True)


        #랜덤 포레스트 회귀 모델 생성
        on_model = RandomForestRegressor(
            n_estimators=200,           # 트리 개수 증가
            max_depth=15,               # 깊이 증가
            min_samples_split=5,        # 분할 최소 샘플 수
            min_samples_leaf=2,         # 리프 노드 최소 샘플 수
            max_features='sqrt',        # 피처 선택 방법
            random_state=42,
            n_jobs=-1,
            oob_score=True              # Out-of-bag 점수 계산
        )

        #  모델 학습
        on_model.fit(X_train, y_train)

        #테스트 데이터 예측
        y_pred = on_model.predict(X_test)

        # 모델 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 성능 지표 출력
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{rmse:.3f}")
        with col2:
            st.metric("MAE", f"{mae:.3f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        with col4:
            st.metric("OOB Score", f"{on_model.oob_score_:.3f}")

        #✅예측 결과 시각화(실제 전환율 VS 예측 전환율 비교)
        fig_ml_on = go.Figure()

        # 실제 값과 예측 값 비교를 위한 산점도 추가
        fig_ml_on.add_trace(go.Scatter(
            x=y_test,         # 실제 값
            y=y_pred,         # 예측 값
            mode='markers+lines',  # 마커와 선을 동시에 표시
            marker=dict(symbol='circle', size=8, color='blue', line=dict(width=2)),
            line=dict(shape='linear'),
            name='예측 vs 실제'  # 레전드에 표시될 이름
        ))

        # 레이아웃 설정
        fig_ml_on.update_layout(
            title='✅전환율 예측 결과 비교',
            xaxis_title='실제 전환율',
            yaxis_title='예측 전환율',
            height=600,
            xaxis=dict(showgrid=True),  # X축 그리드 표시
            yaxis=dict(showgrid=True),  # Y축 그리드 표시
        )

        # Streamlit에서 시각화 표시
        st.plotly_chart(fig_ml_on)
    
        #✅사용자가 입력한 값을 기반으로 전환율 예측
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns = features)
        input_data["체류시간(min)"] = time_input    #선택된 체류 시간 입력

        #선택된 디바이스 및 유입 경로에 대한 원-핫 인코딩 적용
        for device in select_device:
            if f"디바이스_{device}" in input_data.columns:
                input_data[f"디바이스_{device}"] = 1

        for path in select_path:
            if f"유입경로_{path}" in input_data.columns:
                input_data[f"유입경로_{path}"] = 1

        # 입력 데이터도 동일하게 표준화
        input_data_scaled = scaler.transform(input_data)
        input_data_scaled = pd.DataFrame(input_data_scaled, columns=features)

        #전환율 예측 결과 출력
        predicted_conversion = on_model.predict(input_data)[0]

        # 신뢰구간 계산 (부트스트랩 방법)
        predictions = []
        for _ in range(100):
            # 부트스트랩 샘플링
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[indices]
            y_boot = y_train.iloc[indices]
            
            # 부트스트랩 모델 학습
            boot_model = RandomForestRegressor(n_estimators=50, random_state=np.random.randint(1000))
            boot_model.fit(X_boot, y_boot)
            pred = boot_model.predict(input_data_scaled)[0]
            predictions.append(pred)
        
        confidence_lower = np.percentile(predictions, 2.5)
        confidence_upper = np.percentile(predictions, 97.5)
       
        st.subheader(f"예상 전환율 : {predicted_conversion:.2f}%")
        st.write(f"95% 신뢰구간: {confidence_lower:.2f}% ~ {confidence_upper:.2f}%")

with tab5:  #방문자 수 예측
    # 데이터 출력
    with st.expander('오프라인 데이터'):
        st.dataframe(df_off, use_container_width=True)

    city_options = list(city_mapping.values())

    # 학습 데이터 준비
    df_ml_off = df_off.groupby(["날짜", "지역"])["방문자수"].sum().reset_index()
    df_ml_off["날짜"] = pd.to_datetime(df_ml_off["날짜"])

    # 시계열 피처
    df_ml_off["year"] = df_ml_off["날짜"].dt.year
    df_ml_off["month"] = df_ml_off["날짜"].dt.month
    df_ml_off["day"] = df_ml_off["날짜"].dt.day
    df_ml_off["day_of_week"] = df_ml_off["날짜"].dt.weekday
    df_ml_off["quarter"] = df_ml_off["날짜"].dt.quarter
    df_ml_off["week_of_year"] = df_ml_off["날짜"].dt.isocalendar().week
    df_ml_off["is_weekend"] = (df_ml_off["day_of_week"] >= 5).astype(int)

    # 계절성 피처
    df_ml_off["month_sin"] = np.sin(2 * np.pi * df_ml_off["month"] / 12)
    df_ml_off["month_cos"] = np.cos(2 * np.pi * df_ml_off["month"] / 12)
    df_ml_off["day_sin"] = np.sin(2 * np.pi * df_ml_off["day_of_week"] / 7)
    df_ml_off["day_cos"] = np.cos(2 * np.pi * df_ml_off["day_of_week"] / 7)

    select_region = st.selectbox("지역을 선택하세요.", city_options)

    df_region = df_ml_off[df_ml_off["지역"] == select_region]  # 특정 지역 데이터 사용

    # 결측값 처리
    df_region["방문자수"] = df_region["방문자수"].fillna(df_region["방문자수"].median())

    features = ["year", "month", "day", "day_of_week", "quarter", "week_of_year", 
                "is_weekend", "month_sin", "month_cos", "day_sin", "day_cos"]
    X = df_region[features]
    y = df_region["방문자수"]

    if st.button("오프라인 방문자 수 예측"):  # 향후 12개월간의 방문자 수 예측
        scaler_off = StandardScaler()
        X_scaled = scaler_off.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 모델 생성
        off_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

        # 모델 학습
        off_model.fit(X_train, y_train)

        # 모델 성능 평가
        y_pred_test = off_model.predict(X_test)
        mse_off = mean_squared_error(y_test, y_pred_test)
        rmse_off = np.sqrt(mse_off)
        r2_off = r2_score(y_test, y_pred_test)

        # 성능 지표 출력
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse_off:.0f}")
        with col2:
            st.metric("R² Score", f"{r2_off:.3f}")
        with col3:
            st.metric("OOB Score", f"{off_model.oob_score_:.3f}")

        # 최대 날짜의 다음 달부터 12개월 간의 날짜 생성
        max_date = df_region["날짜"].max()
        start_date = (max_date + pd.DateOffset(months=1)).replace(day=1)  # 다음 달의 첫날
        future_dates = pd.date_range(start=start_date, periods=365, freq="D")
        
        future_df = pd.DataFrame({
            "year": future_dates.year,
            "month": future_dates.month,
            "day": future_dates.day,
            "day_of_week": future_dates.weekday,
            "quarter": future_dates.quarter,
            "week_of_year": future_dates.isocalendar().week,
            "is_weekend": (future_dates.weekday >= 5).astype(int),
            "month_sin": np.sin(2 * np.pi * future_dates.month / 12),
            "month_cos": np.cos(2 * np.pi * future_dates.month / 12),
            "day_sin": np.sin(2 * np.pi * future_dates.weekday / 7),
            "day_cos": np.cos(2 * np.pi * future_dates.weekday / 7)
        })

        # 미래 데이터도 표준화
        future_scaled = scaler_off.transform(future_df[features])
        future_scaled = pd.DataFrame(future_scaled, columns=features)
        
        # 방문자 수 예측
        future_pred = off_model.predict(future_scaled)
        future_df["예측 방문자 수"] = np.maximum(future_pred, 0)  # 음수 방지

        # "년-월" 형식의 칼럼 만들기
        future_df["년월"] = future_df["year"].astype(str) + "-" + future_df["month"].astype(str).str.zfill(2)  # 월을 두 자리로 표시

        # 월 별로 집계한 방문자 수
        future_summary = future_df.groupby("년월", as_index=False)["예측 방문자 수"].sum()

        # 예측 방문자 수 형식 변경
        future_summary["예측 방문자 수"] = future_summary["예측 방문자 수"].astype(int).astype(str) + "명"

        st.subheader(f":chart: 향후 12개월 동안 {select_region}의 방문자 수 예측")

        # 방문자 수 예측 시각화
        fig_ml_off = go.Figure()

        # 예측 방문자 수 선 그래프 추가
        fig_ml_off.add_trace(go.Scatter(
            x=future_summary["년월"],
            y=future_summary["예측 방문자 수"].str.extract('(\d+)')[0].astype(int),  # 숫자만 추출하여 y값으로 사용
            mode='markers+lines',  # 마커와 선을 동시에 표시
            marker=dict(symbol='circle', size=8, color='red'),
            line=dict(shape='linear'),
            name='예측 방문자 수'  # 레전드에 표시될 이름
        ))

        # 레이아웃 설정
        fig_ml_off.update_layout(
            title=f"{select_region}의 방문자 수 예측",
            xaxis_title='년-월',
            yaxis_title='방문자 수',
            height=600,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )

        # Streamlit에서 시각화와 데이터프레임 표시
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_ml_off)

        with col2:
            st.dataframe(future_summary[["년월", "예측 방문자 수"]], height=550)

        

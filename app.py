# app.py - Advanced Data Analyst Agent (deploy-ready)
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from agent_core import load_data, profile_data, train_model, forecast_series, cleaning_suggestions, generate_report, html_download_link
from nl_agent import nl_to_action, execute_action
from db_utils import read_sql_table, test_connection
from auth import require_password

@require_password
def secure_app():
    st.set_page_config(page_title='Advanced Data Analyst Agent', layout='wide')
    st.title('Advanced Data Analyst Agent (Secure)')

    # Sidebar - data source
    source = st.sidebar.selectbox('Data source', ['Upload file','Sample CSV','Database'])
    df = None
    if source == 'Upload file':
        uploaded = st.file_uploader('Upload CSV/XLSX', type=['csv','xlsx'])
        if uploaded:
            df = load_data(uploaded)
    elif source == 'Sample CSV':
        sample_path = os.path.join(os.getcwd(), 'sample_data.csv')
        if os.path.exists(sample_path):
            df = load_data(sample_path)
    elif source == 'Database':
        conn = st.sidebar.text_input('Connection string', value='')
        table = st.sidebar.text_input('Table name', value='')
        if st.sidebar.button('Load from DB'):
            if conn and table:
                try:
                    df = read_sql_table(conn, table)
                except Exception as e:
                    st.sidebar.error(f'Failed to load table: {e}')
            else:
                st.sidebar.error('Provide connection string and table name')

    if df is not None:
        st.sidebar.success(f'Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols')
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['Overview','Visualize','Model','Forecast','Cleaning','Chat'])

        with tab1:
            st.subheader('Overview & Profile')
            st.dataframe(df.head())
            p = profile_data(df)
            st.json(p)
            st.subheader('Top numeric summary')
            num = df.select_dtypes(include='number')
            st.dataframe(num.describe().T)

        with tab2:
            st.subheader('Visualize')
            col = st.selectbox('Column', df.columns)
            if pd.api.types.is_numeric_dtype(df[col]):
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}'))
            else:
                st.plotly_chart(px.bar(df[col].value_counts().nlargest(30), title=f'Top categories in {col}'))

            if st.checkbox('Scatter / correlation plot'):
                num_cols = df.select_dtypes(include='number').columns
                xcol = st.selectbox('X axis', num_cols, index=0)
                ycol = st.selectbox('Y axis', num_cols, index=1 if len(num_cols)>1 else 0)
                st.plotly_chart(px.scatter(df, x=xcol, y=ycol, trendline='ols', title=f'{xcol} vs {ycol}'))

        with tab3:
            st.subheader('AutoML')
            target = st.selectbox('Target', df.columns)
            task = st.radio('Task', ['regression','classification'])
            n_iter = st.slider('Hyperparam search iterations', min_value=5, max_value=50, value=20)
            if st.button('Train model'):
                with st.spinner('Training... this may take a while for large datasets'):
                    res = train_model(df, target, task=task, n_iter=n_iter)
                st.success('Model trained')
                st.write('Metrics:')
                st.json(res['metrics'])
                st.subheader('Top feature importances')
                st.dataframe(pd.DataFrame(res['feature_importances'], columns=['feature','importance']).head(30))
                st.session_state['model_result'] = res

            if st.button('Generate report (with model)'):
                mr = st.session_state.get('model_result')
                html = generate_report(df, model_result=mr)
                href, fname = html_download_link(html)
                st.markdown(f'[Download report]({href})', unsafe_allow_html=True)

        with tab4:
            st.subheader('Forecasting')
            date_col = st.selectbox('Date column', df.columns, index=0)
            value_col = st.selectbox('Value column', df.columns, index=1 if len(df.columns)>1 else 0)
            periods = st.number_input('Forecast periods (months)', min_value=1, value=12)
            if st.button('Run forecast'):
                with st.spinner('Forecasting...'):
                    try:
                        forecast = forecast_series(df, date_col, value_col, periods=int(periods))
                        st.write(forecast.head(20))
                        st.line_chart(forecast.set_index('ds')['yhat'])
                    except Exception as e:
                        st.error('Forecast failed: ' + str(e))

        with tab5:
            st.subheader('Cleaning suggestions')
            suggestions = cleaning_suggestions(df)
            st.write(suggestions)

        with tab6:
            st.subheader('Natural-language Chat (requires OPENAI_API_KEY in env)')
            question = st.text_input('Ask a question about the data (e.g., "show average sales per month for region East")')
            if st.button('Ask') and question:
                cols = {c: str(df[c].dtype) for c in df.columns}
                try:
                    action = nl_to_action(question, cols)
                    out = execute_action(df, action)
                    st.write(out)
                    if isinstance(out, dict) and 'plot' in out:
                        d = out['plot']
                        kind = d.get('kind')
                        if kind == 'scatter':
                            st.plotly_chart(px.scatter(df, x=d.get('x'), y=d.get('y')))
                except Exception as e:
                    st.error('NL action failed: ' + str(e))
    else:
        st.info('Upload a CSV/XLSX file, choose Sample CSV, or load from Database (use sidebar).')

if __name__ == '__main__':
    secure_app()

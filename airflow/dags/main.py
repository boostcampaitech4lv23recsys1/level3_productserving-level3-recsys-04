from airflow import DAG
from airflow.operators.python import PythonOperator

from crawling import crawling
from crawling2 import crawling2
from crawling3 import crawling3
from batch_modeling import batch

# from pytz import timezone
from datetime import datetime, timedelta


default_args = {
    'owner' : 'kyle',
    'depends_on_past' : False,
    'start_date' : datetime(2023, 2, 10),
    # 'start_date' : datetime.now(timezone('Asia/Seoul')),
    'retires' : 1,
    'retry_delay' : timedelta(minutes = 15)
}

with DAG(
    dag_id = 'airflow_run',
    default_args = default_args,
    schedule_interval = '@once', #'0 0 * * *'
    tags = ['my_dags']
) as dag:
    
    URLS = [
        "https://map.naver.com/v5/search/%EC%A2%85%EB%A1%9C%EA%B5%AC%20%ED%8F%89%EC%B0%BD%EB%8F%99%20%EC%9D%8C%EC%8B%9D%EC%A0%90?c=14132935.1526363,4524638.2400055,13,0,0,0,dh&isCorrectAnswer=true"
    ]
    crawl_1 = PythonOperator( 
        task_id = 'crawl_1',
        python_callable = crawling,
        op_args = ['Jongno', URLS]
    )
    crawl_2 = PythonOperator(
        task_id = 'crawl_2',
        python_callable = crawling2,
        op_args = ['Jongno']
    )
    crawl_3 = PythonOperator(
        task_id = 'crawl_3',
        python_callable = crawling3,
        op_args = ['Jongno']
    )
    model_train = PythonOperator(
        task_id = 'model_train',
        python_callable = batch,
    )
    crawl_1 >> crawl_2 >> crawl_3 >> model_train

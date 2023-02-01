from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from crawling import crawling
from crawling2 import crawling2
from crawling3 import crawling3

default_args = {
    'owner' : 'kyle',
    'depends_on_past' : False,
    'start_date' : datetime(2023, 1, 27),
    'retires' : 1,
    'retry_delay' : timedelta(minutes = 15)
}

with DAG(
    dag_id = 'crawling1',
    default_args = default_args,
    schedule_interval = '@once', #'0 0 * * *'
    tags = ['my_dags']
) as dag:
    
    URLS = [
        "https://map.naver.com/v5/search/%EC%A2%85%EB%A1%9C%EA%B5%AC%20%ED%8F%89%EC%B0%BD%EB%8F%99%20%EC%9D%8C%EC%8B%9D%EC%A0%90?c=14132935.1526363,4524638.2400055,13,0,0,0,dh&isCorrectAnswer=true",  # 평창동
    ]
    task1 = PythonOperator( 
        task_id = 'task1',
        python_callable = crawling,
        op_args = ['Jongno', URLS]
    )
    
    task2 = PythonOperator(
        task_id = 'task2',
        python_callable = crawling2,
        op_args = ['Jongno']
    )
    
    task3 = PythonOperator(
        task_id = 'task3',
        python_callable = crawling3,
        op_args = ['Jongno']
    )
    
    task1 >> task2 >> task3
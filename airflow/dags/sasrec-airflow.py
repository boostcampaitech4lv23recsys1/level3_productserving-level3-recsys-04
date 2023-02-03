from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from batch_modeling import batch

default_args = {
    'owner' : 'kyle',
    'depends_on_past' : False,
    'start_date' : datetime(2023, 1, 31),
    'retires' : 1,
    'retry_delay' : timedelta(minutes = 15)
}

with DAG(
    dag_id = 'sasrec',
    default_args = default_args,
    schedule_interval = '@once', #'0 0 * * *'
    tags = ['my_dags']
) as dag:
    
    task1 = PythonOperator( 
        task_id = 'task1',
        python_callable = batch
    )
    
    task1
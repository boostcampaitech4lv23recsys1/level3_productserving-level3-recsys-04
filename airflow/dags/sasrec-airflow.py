from airflow import DAG
from airflow.operators.python import PythonOperator
from batch_modeling_temp import batch
from datetime import date, datetime, timedelta


default_args = {
    # 'owner' : 'kyle',
    'depends_on_past' : False,
    'start_date' : datetime(date.today().year, date.today().month, date.today().day),
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

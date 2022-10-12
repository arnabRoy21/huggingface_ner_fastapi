from fastapi import FastAPI


app = FastAPI()

@app.get('/')
def index():
    return {'data': 'hello'}

@app.get('/about')
def about():
    return {'data': {'name': 'arnab'}}
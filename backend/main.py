from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def index():
    return {'Name':'Linh'}


@app.get('/items/{item_id}')
def get_item(item_id: str):
    a = item_id.lower()
    return {'item_id':item_id}

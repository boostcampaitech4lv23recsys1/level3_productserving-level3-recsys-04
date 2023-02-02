if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=7777, reload=True)

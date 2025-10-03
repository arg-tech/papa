from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from . import papa


class RequestBody(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    xaif: dict


app = FastAPI()


@app.post("/api/all_analytics")
# Call without async so that fastapi does the work on a threadpool as
# all_analytics is cpu-bound and we don't want to block the current thread.
def all_analytics(body: RequestBody | dict):
    kwargs = {}
    if isinstance(body, RequestBody):
        xaif = body.xaif
        for name, value in body:
            if name == "xaif":
                continue

            kwargs[name] = value if value is not None else False
    else:
        xaif = body

    return papa.all_analytics(xaif, **kwargs)

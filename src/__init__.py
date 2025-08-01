from fastapi import FastAPI
from pydantic import BaseModel
from . import papa


# TODO: Flesh this out?
class Xaif(BaseModel):
    AIF: dict
    text: str
    OVA: dict | None = None


class RequestBody(BaseModel):
    xaif: Xaif
    node_level: bool | None = None
    speaker: bool | None = None
    forecast: bool | None = None


app = FastAPI()


@app.post("/api/all_analytics")
# Call without async so that fastapi does the work on a threadpool as
# all_analytics is cpu-bound and we don't want to block the current thread.
def all_analytics(body: RequestBody | Xaif):
    if isinstance(body, RequestBody):
        xaif = dict(body.xaif)
        kwargs = {}
        for name, value in body:
            if name == "xaif":
                continue

            kwargs[name] = value if value is not None else False
    else:
        xaif = dict(body)

    return papa.all_analytics(xaif)

# papa
Papa: Amazing Python Analytics (in progress)

## Web deployment

Papa can be deployed as a web server using the provided Dockerfile.

### Usage

A live version with a web ui is available at http://papa.amfws.arg.tech/.

The api endpoint is available at http://papa.amfws.arg.tech/api/all_analytics.
It expects either an argument map, or an object containing an argument map
alongside any of the three optional parameters that papa's all_analytics
function takes, always in json format.

Example:
```bash
curl --data '@18841.json' --header 'Content-Type: application/json' http://papa.amfws.arg.tech/api/all_analytics -o out.json
```
Note: The "@" on the beginning of the filename is not part of the filename, it's
to tell curl that a filename is next.

If you want to send arguments to the all_analytics function, it may be easier to use a programming language, such as Python.

For an example in Python, add the args and the argument map to a dict, as shown on the
`json=...` line.
```py
import requests
import json

with open("18841.json", "r") as f:
    data = json.load(f)

response = requests.post(
    "http://papa.amfws.arg.tech/api/all_analytics",
    json={"xaif": data, "node_level": True, "speaker": True, "forecast": True},
)

with open("out.json", "w") as f:
    json.dump(response.json(), f)
```

from django.http import HttpResponse
from . import predict
import json


def upload_file(request):
    if request.method == "POST":
        myFile = request.FILES.get("file", None)
        data = myFile.read()
        res, anm, vhc = predict.get_result(data)
        if res:
            result = json.dumps(
                {
                    'code': 0,
                    'data':
                        {
                            'anm': str(anm * 100) + '%',
                            'vhc': str(vhc * 100) + '%'
                        }
                })
            return HttpResponse(result, content_type="application/json")
    return HttpResponse(json.dumps({'code': 1}), content_type="application/json")

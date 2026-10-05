import httpx
import respx

from eis.ingest._http import fetch_json


@respx.mock
def test_fetch_json_returns_parsed_body_on_success():
    respx.get("https://example.com/api").mock(return_value=httpx.Response(200, json={"a": 1}))
    assert fetch_json("https://example.com/api") == {"a": 1}


@respx.mock
def test_fetch_json_returns_none_on_http_error():
    respx.get("https://example.com/api").mock(return_value=httpx.Response(500))
    assert fetch_json("https://example.com/api") is None


@respx.mock
def test_fetch_json_returns_none_on_malformed_json():
    respx.get("https://example.com/api").mock(return_value=httpx.Response(200, text="not json"))
    assert fetch_json("https://example.com/api") is None


@respx.mock
def test_fetch_json_passes_params_and_headers():
    route = respx.get("https://example.com/api").mock(return_value=httpx.Response(200, json={}))
    fetch_json("https://example.com/api", params={"q": "x"}, headers={"User-Agent": "test"})
    request = route.calls.last.request
    assert request.url.params["q"] == "x"
    assert request.headers["User-Agent"] == "test"

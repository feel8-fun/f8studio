from __future__ import annotations

import ast
from pathlib import Path

from fastapi.routing import APIRoute

from f8studio_server import app as app_module
from f8studio_server.api_contracts import ROUTES
from f8studio_server.app import create_app
from f8studio_server.application import StudioApplication


def test_every_http_api_has_a_contract_and_openapi_references_resolve(tmp_path: Path) -> None:
    studio = StudioApplication(data_dir=tmp_path / 'data', service_roots=())
    app = create_app(web_dist=tmp_path, application=studio)
    actual = {(route.path, method.lower()) for route in app.routes if isinstance(route, APIRoute)
              and route.path.startswith('/api/') for method in route.methods}
    declared = {(route.path, route.method) for route in ROUTES}
    assert actual == declared
    assert len(declared) == len(ROUTES)
    schema = app.openapi()
    components = schema['components']['schemas']

    def verify(value: object) -> None:
        if isinstance(value, dict):
            if '$ref' in value:
                prefix = '#/components/schemas/'
                assert value['$ref'].startswith(prefix)
                assert value['$ref'][len(prefix):] in components
            for child in value.values():
                verify(child)
        elif isinstance(value, list):
            for child in value:
                verify(child)

    verify(schema)
    for route in ROUTES:
        operation = schema['paths'][route.path][route.method]
        response = operation['responses'][str(route.status)]
        assert ('content' in response) == (route.response is not None)
        assert ('requestBody' in operation) == (route.request is not None)
    patch = schema['paths']['/api/projects/{project_id}/patch']['post']
    assert patch['requestBody']['content']['application/json']['schema']['$ref'].endswith('/PatchRequestInput')
    assert patch['responses']['200']['content']['application/json']['schema']['$ref'].endswith('/PatchResult')
    studio.editor.close()


def test_contract_requests_match_actual_decoders() -> None:
    # Ensure explicit documentation cannot silently diverge from route decoding.
    tree = ast.parse(Path(app_module.__file__).read_text())
    contracts = {(route.path, route.method): route for route in ROUTES}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        decoders = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and call.func.id == '_decode_body']
        if not decoders:
            continue
        assert len(decoders) == 1
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            route = contracts[(ast.literal_eval(decorator.args[0]), decorator.func.attr)]
            assert route.request.__name__ == ast.unparse(decoders[0].args[1])

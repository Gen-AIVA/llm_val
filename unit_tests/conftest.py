import os
import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption('--contour',
                     action='store',
                     default='mlspace',
                     help="contour option. Allowed parameters: ['sigma', 'mlspace', 'datalab']")


@pytest.fixture(scope='session')
def oauth_url(request, pytestconfig):
    option = pytestconfig.getoption('--contour')
    URLS = {'mlspace': 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
            'sigma': 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth',
            'datalab': None}
    if option not in URLS:
        pytest.skip(f'Invalid parameter: {option}. Allowed parameters are: [{", ".join(list(URLS.keys()))}]')
    return URLS[option]


@pytest.fixture(scope='session')
def api_url(request, pytestconfig):
    option = pytestconfig.getoption('--contour')
    URLS = {'mlspace': 'https://gigachat.devices.sberbank.ru/api/v1',
            'sigma': 'https://gigachat.devices.sberbank.ru/api/v1',
            'datalab': 'http://liveaccess/v1/gc'}
    if option not in URLS:
        pytest.skip(f'Invalid parameter: {option}. Allowed parameters are: [{", ".join(list(URLS.keys()))}]')
    return URLS[option]


def pytest_generate_tests(metafunc):
    option = metafunc.config.getoption('--contour')
    if 'api_param' in metafunc.fixturenames:
        metafunc.parametrize('api_param', API_PARAMS[option])
    if 'api_single_param' in metafunc.fixturenames:
        metafunc.parametrize('api_single_param', API_SINGLE_PARAMS[option])


AUTH_DATA = 'INSERT_YOUR_CLIENT_SECRET_HERE'
API_PARAMS = {'mlspace': [({'contour': 'mlspace', 
                            'token': None, 
                            'auth_data': AUTH_DATA,
                            'api_url': None,
                            'oauth_url': None}), 
                          ({'contour': None, 
                            'token': None, 
                            'auth_data': AUTH_DATA,
                            'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                            'oauth_url': 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'}), 
                          ({'contour': 'mlspace', 
                            'token': None, 
                            'auth_data': AUTH_DATA,
                            'api_url': None,
                            'oauth_url': 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'}),
                          ({'contour': 'mlspace', 
                            'token': None, 
                            'auth_data': AUTH_DATA,
                            'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                            'oauth_url': None})],
              'sigma': [({'contour': 'sigma',
                          'token': None,
                          'auth_data': AUTH_DATA,
                          'api_url': None,
                          'oauth_url': None}), 
                        ({'contour': None,
                          'token': None,
                          'auth_data': AUTH_DATA,
                          'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                          'oauth_url': 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth'}), 
                        ({'contour': 'mlspace',
                          'token': None,
                          'auth_data': AUTH_DATA,
                          'api_url': None,
                          'oauth_url': 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth'}),
                        ({'contour': 'sigma', 
                          'token': None, 
                          'auth_data': AUTH_DATA,
                          'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                          'oauth_url': None})],
              'datalab': [({'contour': 'datalab', 
                            'token': os.environ.get('JPY_API_TOKEN'), 
                            'auth_data': None,
                            'api_url': None,
                            'oauth_url': None}), 
                          ({'contour': None, 
                            'token': os.environ.get('JPY_API_TOKEN'),
                            'auth_data': None,
                            'api_url': 'http://liveaccess/v1/gc',
                            'oauth_url': None}), 
                          ({'contour': 'datalab', 
                            'token': os.environ.get('JPY_API_TOKEN'),
                            'auth_data': None,
                            'api_url': 'http://liveaccess/v1/gc',
                            'oauth_url': None})]}

API_SINGLE_PARAMS = {'mlspace': [{'contour': 'mlspace', 
                                 'token': None, 
                                 'auth_data': AUTH_DATA,
                                 'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                                 'oauth_url': 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'}],
                    'sigma': [{'contour': 'sigma',
                              'token': None,
                              'auth_data': AUTH_DATA,
                              'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                              'oauth_url': 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth'}],
                    'datalab': [{'contour': 'datalab', 
                                'token': os.environ.get('JPY_API_TOKEN'),
                                'auth_data': None,
                                'api_url': 'http://liveaccess/v1/gc',
                                'oauth_url': None}]}


RAISE_API_PARAMS = [({'contour': None, 
                      'token': None, 
                      'auth_data': None,
                      'api_url': None,
                      'oauth_url': None}),
                    ({'contour': 'Hello, World!', 
                      'token': None, 
                      'auth_data': None,
                      'api_url': None,
                      'oauth_url': None}), 
                    ({'contour': None, 
                      'token': None, 
                      'auth_data': None,
                      'api_url': None,
                      'oauth_url': 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'}),
                   ({'contour': None, 
                      'token': None, 
                      'auth_data': None,
                      'api_url': 'https://gigachat.devices.sberbank.ru/api/v1',
                      'oauth_url': None})]
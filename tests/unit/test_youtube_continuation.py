from avtdl.plugins.youtube.common import get_continuation_api_url, prepare_next_page_request


def test_prepare_next_page_request_uses_search_api_url():
    url, headers, body = prepare_next_page_request(
        {'client': {'clientVersion': '1.2.3', 'visitorData': 'visitor'}},
        'token',
        api_url='/youtubei/v1/search',
    )

    assert url.startswith('https://www.youtube.com/youtubei/v1/search?key=')
    assert headers['X-Youtube-Client-Version'] == '1.2.3'
    assert body['continuation'] == 'token'


def test_prepare_next_page_request_defaults_to_browse():
    url, _, _ = prepare_next_page_request({}, 'token')

    assert url.startswith('https://www.youtube.com/youtubei/v1/browse?key=')


def test_get_continuation_api_url_from_command_metadata():
    api_url = get_continuation_api_url({
        'commandMetadata': {
            'webCommandMetadata': {
                'apiUrl': '/youtubei/v1/search',
            },
        },
        'continuationCommand': {
            'token': 'token',
        },
    })

    assert api_url == '/youtubei/v1/search'

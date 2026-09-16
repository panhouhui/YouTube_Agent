import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import Field, PositiveFloat

from avtdl.core.interfaces import Record
from avtdl.core.monitors import PagedFeedMonitor, PagedFeedMonitorConfig, PagedFeedMonitorEntity
from avtdl.core.plugins import Plugins
from avtdl.core.request import HttpClient, RetrySettings
from avtdl.core.runtime import RuntimeContext
from avtdl.core.utils import find_all, find_one
from avtdl.plugins.mattermost.mattermost import load_env_file
from avtdl.plugins.youtube.common import (
    NextPageContext,
    extract_keys,
    get_continuation_token,
    get_innertube_context,
    get_session_index,
    handle_consent,
    parse_navigation_endpoint,
    prepare_next_page_request,
)
from avtdl.plugins.youtube.feed_info import AuthorInfo, parse_owner_info


POST_URL_PREFIX = 'https://www.youtube.com/post/'


@Plugins.register('youtube.community', Plugins.kind.ASSOCIATED_RECORD)
class YoutubeCommunityPostRecord(Record):
    post_id: str
    url: str
    source_url: str
    author: Optional[str] = None
    author_url: Optional[str] = None
    avatar_url: Optional[str] = None
    content: str = ''
    published_text: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    image_urls_text: str = ''

    def __str__(self) -> str:
        return f'{self.author or "YouTube"}\n{self.published_text or ""}\n{self.content}\n{self.url}'

    def __repr__(self) -> str:
        content = re.sub(r'\s+', ' ', self.content).strip()
        return f'YoutubeCommunityPostRecord({self.post_id}, {content[:60]})'

    def get_uid(self) -> str:
        return self.post_id

    def as_embed(self) -> Dict[str, Any]:
        embed: Dict[str, Any] = {
            'title': self.author or 'YouTube community post',
            'description': self.content,
            'url': self.url,
            'author': {'name': self.author, 'url': self.author_url, 'icon_url': self.avatar_url},
            'footer': {'text': self.published_text or ''},
        }
        if self.image_urls:
            embed['image'] = {'url': self.image_urls[0]}
        return embed


@Plugins.register('youtube.community', Plugins.kind.ACTOR_CONFIG)
class YoutubeCommunityMonitorConfig(PagedFeedMonitorConfig):
    pass


@Plugins.register('youtube.community', Plugins.kind.ACTOR_ENTITY)
class YoutubeCommunityMonitorEntity(PagedFeedMonitorEntity):
    update_interval: PositiveFloat = 300
    env_file: Path = Path('.env')
    """path to .env containing YouTube_cookie"""
    cookie_env: str = 'YouTube_cookie'
    """env key containing a raw YouTube Cookie header value"""


class CommunityPageContext(NextPageContext):
    owner_info: Optional[AuthorInfo] = None


def text_from_runs(value: Any) -> str:
    if not isinstance(value, dict):
        return ''
    simple = value.get('simpleText')
    if isinstance(simple, str):
        return simple.strip()
    runs = value.get('runs')
    if not isinstance(runs, list):
        return ''
    parts: List[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        text = run.get('text')
        if isinstance(text, str):
            parts.append(text)
    return ''.join(parts).strip()


def post_url(post: dict, post_id: str) -> str:
    endpoint = find_one(post, '$..commandMetadata.webCommandMetadata.url')
    if isinstance(endpoint, str) and endpoint.startswith('/post/'):
        return 'https://www.youtube.com' + endpoint
    return POST_URL_PREFIX + post_id


def author_url(post: dict, owner_info: Optional[AuthorInfo]) -> Optional[str]:
    url = find_one(post, '$.authorEndpoint.commandMetadata.webCommandMetadata.url')
    if isinstance(url, str):
        if url.startswith('/'):
            return 'https://www.youtube.com' + url
        return url
    if owner_info is not None:
        return owner_info.channel
    return None


def external_links(post: dict) -> List[str]:
    links: List[str] = []
    runs = find_all(post, '$.contentText.runs')
    for run_group in runs:
        if not isinstance(run_group, list):
            continue
        for run in run_group:
            if not isinstance(run, dict):
                continue
            endpoint = run.get('navigationEndpoint')
            if not isinstance(endpoint, dict):
                continue
            try:
                link = parse_navigation_endpoint(endpoint)
            except Exception:
                continue
            if link.startswith('http') and link not in links:
                links.append(link)
    return links


def parse_post(post: dict, owner_info: Optional[AuthorInfo], source_url: str) -> Optional[YoutubeCommunityPostRecord]:
    post_id = post.get('postId')
    if not isinstance(post_id, str) or not post_id:
        return None

    content = text_from_runs(post.get('contentText'))
    links = external_links(post)
    if links:
        content = content.rstrip() + '\n\nLinks:\n' + '\n'.join(links)

    author = text_from_runs(post.get('authorText'))
    if not author and owner_info is not None:
        author = owner_info.name

    images = find_all(post, '$..backstageImageRenderer.image.thumbnails.[::-1].url')
    image_urls = [str(url) for url in images if isinstance(url, str)]
    avatar = find_one(post, '$.authorThumbnail.thumbnails.[::-1].url')
    if not isinstance(avatar, str) and owner_info is not None:
        avatar = owner_info.avatar_url

    published = text_from_runs(post.get('publishedTimeText'))
    return YoutubeCommunityPostRecord(
        post_id=post_id,
        url=post_url(post, post_id),
        source_url=source_url,
        author=author or None,
        author_url=author_url(post, owner_info),
        avatar_url=avatar,
        content=content,
        published_text=published or None,
        image_urls=image_urls,
        image_urls_text='\n'.join(image_urls),
    )


@Plugins.register('youtube.community', Plugins.kind.ACTOR)
class YoutubeCommunityMonitor(PagedFeedMonitor):
    """Monitor YouTube community posts, such as https://www.youtube.com/@name/posts."""

    def __init__(self, conf: YoutubeCommunityMonitorConfig, entities: Sequence[YoutubeCommunityMonitorEntity],
                 ctx: RuntimeContext):
        super().__init__(conf, entities, ctx)
        logger = logging.getLogger('actor').getChild(conf.name)
        for entity in entities:
            env = load_env_file(entity.env_file, logger)
            cookie = env.get(entity.cookie_env)
            if not cookie:
                logger.info(f'[{entity.name}] no YouTube cookie configured in "{entity.env_file}"')
                continue
            headers = dict(entity.headers or {})
            headers['Cookie'] = cookie
            entity.headers = headers
            logger.info(f'[{entity.name}] loaded YouTube cookie from "{entity.env_file}"')

    async def handle_first_page(self, entity: PagedFeedMonitorEntity, client: HttpClient) -> Tuple[Optional[Sequence[Record]], Optional[CommunityPageContext]]:
        raw_page_text = await self.request(entity.url, entity, client)
        if raw_page_text is None:
            return None, None
        raw_page_text = await handle_consent(raw_page_text, entity.url, client, self.logger)
        posts, continuation_token, page = self.parse_page(raw_page_text, anchor='var ytInitialData = ')
        owner_info = parse_owner_info(page)
        records = self.parse_records(posts, owner_info, entity.url)
        context = CommunityPageContext(
            innertube_context=get_innertube_context(raw_page_text),
            session_index=get_session_index(page),
            continuation_token=continuation_token,
            owner_info=owner_info,
        )
        return records, context

    async def handle_next_page(self, entity: PagedFeedMonitorEntity, client: HttpClient,
                               context: Optional[CommunityPageContext]) -> Tuple[Optional[Sequence[Record]], Optional[CommunityPageContext]]:
        if context is None or context.continuation_token is None:
            return [], None

        url, headers, post_body = prepare_next_page_request(
            context.innertube_context,
            context.continuation_token,
            cookies=client.cookie_jar,
            session_index=context.session_index,
        )
        raw_page = await client.request_text(
            url,
            method='POST',
            data_json=post_body,
            headers=headers,
            settings=RetrySettings(retry_times=3, retry_delay=5, retry_multiplier=2),
        )
        if raw_page is None:
            return None, None

        posts, continuation_token, _ = self.parse_page(raw_page, anchor='')
        records = self.parse_records(posts, context.owner_info, entity.url)
        context.continuation_token = continuation_token
        return records, context if continuation_token is not None else None

    def parse_page(self, raw_page: str, anchor: str) -> Tuple[List[dict], Optional[str], dict]:
        items, page = extract_keys(raw_page, ['backstagePostRenderer', 'continuationEndpoint'], anchor)
        posts = [item for item in items.get('backstagePostRenderer', []) if isinstance(item, dict)]
        continuation_token = get_continuation_token(items.get('continuationEndpoint', []))
        return posts, continuation_token, page

    def parse_records(self, posts: Sequence[dict], owner_info: Optional[AuthorInfo], source_url: str) -> List[YoutubeCommunityPostRecord]:
        records: List[YoutubeCommunityPostRecord] = []
        for post in posts:
            try:
                record = parse_post(post, owner_info, source_url)
            except Exception as e:
                self.logger.warning(f'failed to parse YouTube community post: {type(e).__name__} {e}')
                continue
            if record is not None:
                records.append(record)
        records = records[::-1]
        return records

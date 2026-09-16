from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import shorten
from typing import Dict, Literal, Optional, Sequence

from pydantic import Field

from avtdl.core.actors import Action, ActionEntity, ActorConfig
from avtdl.core.interfaces import Record
from avtdl.core.plugins import Plugins


CHINA_TZ = timezone(timedelta(hours=8))


def now_china() -> str:
    return datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S UTC+08:00')


def clean_text(value: object, fallback: str = '未获取') -> str:
    text = str(value or '').strip()
    return text or fallback


def compact(value: object, width: int = 140, fallback: str = '未获取') -> str:
    text = clean_text(value, fallback)
    return shorten(text.replace('\n', ' '), width=width, placeholder='...')


def normalize_account(value: object) -> str:
    text = str(value or '').strip().casefold()
    for prefix in ('https://www.youtube.com/', 'http://www.youtube.com/', 'https://youtube.com/', 'http://youtube.com/'):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip('/').removeprefix('@')


def load_key_accounts(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        return set()
    accounts: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        accounts.add(normalize_account(line))
    return accounts


@Plugins.register('youtube.keyword_agents', Plugins.kind.ASSOCIATED_RECORD)
class YouTubeAgentRecord(Record):
    agent_role: str
    agent_name: str
    report_id: str
    run_time: str
    scope: str
    source_platform: str = 'YouTube'

    title: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    channel_link: Optional[str] = None
    url: str
    source_url: Optional[str] = None
    video_id: Optional[str] = None
    post_id: Optional[str] = None
    published_text: Optional[str] = None
    matched_keywords: str
    push_reason: str
    ai_confidence: float = 0
    risk_level: str = '关注'

    raw_collected_rows: int = 1
    valid_lead_count: int = 1
    candidate_count: int = 1
    top_keywords: str = '未明确返回'
    related_accounts: str = '未获取'
    candidate_link_summaries: str = ''
    collector_conclusion: str = ''
    collector_note: str = ''

    account_count: int = 0
    new_post_count: int = 0
    failed_count: int = 0
    success_count: int = 0
    state_file_count: int = 0
    duplicate_filtered_count: int = 0
    failed_accounts: str = '无'
    active_accounts: str = '未配置'
    key_event_link_summaries: str = ''
    key_account_conclusion: str = ''
    key_account_action: str = ''

    main_topics: str = ''
    needs_analysis_count: int = 1
    hotspot_conclusion: str = ''
    hotspot_reason: str = ''
    evidence_summary: str = ''
    hotspot_action: str = ''

    executive_summary: str = ''
    key_findings: str = ''
    impact_assessment: str = ''
    analyst_action: str = ''

    def __str__(self) -> str:
        return f'{self.agent_name}\n{self.title or self.main_topics}\n{self.url}'

    def __repr__(self) -> str:
        return f'YouTubeAgentRecord({self.agent_role}, {self.video_id or self.post_id or self.url})'

    def get_uid(self) -> str:
        return f'{self.agent_role}:{self.video_id or self.post_id or self.url}'


@Plugins.register('youtube.keyword_agents', Plugins.kind.ACTOR_CONFIG)
class YouTubeKeywordAgentsConfig(ActorConfig):
    pass


@Plugins.register('youtube.keyword_agents', Plugins.kind.ACTOR_ENTITY)
class YouTubeKeywordAgentEntity(ActionEntity):
    agent_role: Literal['collector', 'key_accounts', 'hotspots', 'analyst']
    scope: str = '普通 YouTube 关键词监控'
    key_accounts_file: Path = Path('key_accounts.youtube.txt')
    """one YouTube handle, channel name, or channel URL per line; only used by key_accounts role"""


@Plugins.register('youtube.keyword_agents', Plugins.kind.ACTOR)
class YouTubeKeywordAgentsAction(Action):
    def __init__(self, conf: YouTubeKeywordAgentsConfig, entities: Sequence[YouTubeKeywordAgentEntity], ctx):
        super().__init__(conf, entities, ctx)
        self.conf: YouTubeKeywordAgentsConfig
        self.entities: Dict[str, YouTubeKeywordAgentEntity]  # type: ignore
        self.key_accounts = {
            entity.name: load_key_accounts(entity.key_accounts_file)
            for entity in entities
        }

    def handle(self, entity: YouTubeKeywordAgentEntity, record: Record):
        if entity.agent_role == 'key_accounts' and not self.is_key_account(entity, record):
            self.logger.debug(f'[{entity.name}] record is not from a configured key account, skipping')
            return
        self.on_record(entity, self.build_agent_record(entity, record))

    def is_key_account(self, entity: YouTubeKeywordAgentEntity, record: Record) -> bool:
        accounts = self.key_accounts.get(entity.name, set())
        if not accounts:
            return False
        data = record.model_dump()
        candidates = {
            normalize_account(data.get('author')),
            normalize_account(data.get('channel_link')),
            normalize_account(data.get('source_url')),
        }
        candidates.discard('')
        return bool(accounts.intersection(candidates))

    def build_agent_record(self, entity: YouTubeKeywordAgentEntity, record: Record) -> YouTubeAgentRecord:
        data = record.model_dump()
        title = data.get('title')
        author = data.get('author')
        url = clean_text(data.get('url'), '未获取原始链接')
        matched = clean_text(data.get('matched_keywords'), '未明确返回')
        reason = clean_text(data.get('push_reason'), '已命中关键词，建议人工复核原始链接。')
        confidence = float(data.get('ai_confidence') or 0)
        risk_level = self.risk_level(data, confidence)
        run_time = now_china()
        link_summary = f'- {compact(title, 90)}｜{clean_text(author)}｜{url}'
        report_id = f'yt-{entity.agent_role}-{data.get("video_id") or data.get("post_id") or record.hash()[:12]}'

        base = dict(
            agent_role=entity.agent_role,
            agent_name=self.agent_name(entity.agent_role),
            report_id=report_id,
            run_time=run_time,
            scope=entity.scope,
            title=title,
            summary=data.get('summary'),
            author=author,
            channel_link=data.get('channel_link'),
            url=url,
            source_url=data.get('source_url'),
            video_id=data.get('video_id'),
            post_id=data.get('post_id'),
            published_text=data.get('published_text'),
            matched_keywords=matched,
            push_reason=reason,
            ai_confidence=confidence,
            risk_level=risk_level,
            top_keywords=matched,
            related_accounts=clean_text(author),
            candidate_link_summaries=link_summary,
            evidence_summary=f'标题：{compact(title)}\n频道/作者：{clean_text(author)}\n判断依据：{reason}',
            main_topics=compact(title or matched),
            hotspot_reason=reason,
            executive_summary=f'发现 1 条普通 YouTube 关键词监控线索，风险等级：{risk_level}。',
            key_findings=f'- 命中词：{matched}\n- 频道/作者：{clean_text(author)}\n- 原始发布时间：{clean_text(data.get("published_text"))}\n- 链接：{url}',
            impact_assessment='当前为单条可复核线索，尚未形成跨视频、跨频道或时间序列传播结论。',
            analyst_action='建议人工打开原始链接复核内容，并继续观察是否出现同议题重复传播。',
        )

        role = entity.agent_role
        if role == 'collector':
            base.update(
                collector_conclusion='该记录通过关键词命中、去重和内容风险分析，已进入后续研判。',
                collector_note='本条快报只使用程序采集到的 YouTube 字段与分析结果；未生成额外来源、数量或趋势。',
            )
        elif role == 'key_accounts':
            base.update(
                account_count=1,
                new_post_count=1,
                success_count=1,
                state_file_count=1,
                active_accounts=clean_text(author),
                key_event_link_summaries=link_summary,
                key_account_conclusion='该记录来自已配置的 YouTube 重点账号清单。',
                key_account_action='建议复核该账号近期连续动态，并确认是否需要加入更高频监控。',
            )
        elif role == 'hotspots':
            base.update(
                hotspot_conclusion='当前仅形成可关注候选，热点基线积累中。',
                hotspot_action='继续观察后续采集是否出现相同标题、相同关键词或同频道重复扩散。',
            )
        elif role == 'analyst':
            base.update(
                needs_analysis_count=1,
            )

        return YouTubeAgentRecord(**base)

    @staticmethod
    def agent_name(role: str) -> str:
        names = {
            'collector': 'YT采集Agent',
            'key_accounts': 'YT重点账号Agent',
            'hotspots': 'YT热点Agent',
            'analyst': 'YT分析Agent',
        }
        return names.get(role, role)

    @staticmethod
    def risk_level(data: Dict[str, object], confidence: float) -> str:
        explicit = clean_text(data.get('ai_risk_level'), '')
        if explicit:
            return explicit
        if confidence >= 0.85:
            return '高'
        if confidence >= 0.7:
            return '中'
        return '关注'

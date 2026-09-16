import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib import request

from pydantic import Field

from avtdl.core.actions import QueueAction, QueueActionConfig, QueueActionEntity
from avtdl.core.interfaces import Record
from avtdl.core.plugins import Plugins
from avtdl.core.request import HttpClient
from avtdl.core.runtime import RuntimeContext
from avtdl.plugins.filters.filters import load_patterns_file
from avtdl.plugins.mattermost.mattermost import load_env_file


DEFAULT_API_URL = 'https://api.minimax.io/v1'
DEFAULT_MODEL = 'MiniMax-M2.7'
CHINA_TZ = timezone(timedelta(hours=8))

APEC_RISK_LEVELS = {
    '一般': '🟢 一般',
    '关注': '🟡 关注',
    '重要': '🟠 重要',
    '重大': '🔴 重大',
}
FORBIDDEN_ANALYSIS_TERMS = (
    'minimax',
    'chatgpt',
    'openai',
    'claude',
    'gemini',
    'deepseek',
    'qwen',
    'gpt',
    '通义',
    '豆包',
)


def minimax_api_key_for(env: Dict[str, str], api_url: str) -> str:
    if 'api.minimax.io' in api_url.casefold():
        return env.get('MINIMAX_INTL_API_KEY') or env.get('MINIMAX_API_KEY', '')
    return env.get('MINIMAX_API_KEY') or env.get('MINIMAX_INTL_API_KEY', '')


def minimax_trust_env_for(env: Dict[str, str], api_url: str) -> bool:
    configured = env.get('MINIMAX_TRUST_ENV')
    if configured is not None:
        return configured.strip().casefold() in {'1', 'true', 'yes', 'on'}
    return 'api.minimax.io' not in api_url.casefold()


def chat_url(api_url: str) -> str:
    api_url = api_url.rstrip('/')
    if api_url.endswith('/chat/completions') or api_url.endswith('/text/chatcompletion_v2'):
        return api_url
    return f'{api_url}/chat/completions'


def extract_message(payload: Dict[str, Any]) -> str:
    base_resp = payload.get('base_resp')
    if isinstance(base_resp, dict) and base_resp.get('status_code') not in {None, 0}:
        raise RuntimeError(f'MiniMax API error: {base_resp}')
    error = payload.get('error')
    if isinstance(error, dict):
        raise RuntimeError(f'MiniMax API error: {error}')
    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        message = choices[0].get('message')
        if isinstance(message, dict) and isinstance(message.get('content'), str):
            return message['content']
        text = choices[0].get('text')
        if isinstance(text, str):
            return text
    reply = payload.get('reply')
    if isinstance(reply, str):
        return reply
    return json.dumps(payload, ensure_ascii=False)


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text.removeprefix('json').strip()
    start = text.find('{')
    if start > 0:
        text = text[start:]
    data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError('MiniMax response JSON is not an object')
    return data


def record_text(record: Record, limit: int = 1800) -> str:
    data = record.model_dump()
    interesting = {
        'title': data.get('title'),
        'summary': data.get('summary'),
        'content': data.get('content'),
        'author': data.get('author'),
        'published_text': data.get('published_text'),
        'url': data.get('url'),
        'source_url': data.get('source_url'),
    }
    text = json.dumps(interesting, ensure_ascii=False, default=str)
    return text[:limit]


def matched_keywords(record: Record, patterns: Sequence[str]) -> list[str]:
    text = record_text(record, limit=10000).casefold()
    found: list[str] = []
    for pattern in patterns:
        if pattern.casefold() in text and pattern not in found:
            found.append(pattern)
    return found


@Plugins.register('minimax.analyze', Plugins.kind.ASSOCIATED_RECORD)
class MiniMaxAnalysisRecord(Record):
    title: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    channel_link: Optional[str] = None
    url: str
    source_url: Optional[str] = None
    video_id: Optional[str] = None
    post_id: Optional[str] = None
    published_text: Optional[str] = None
    content: Optional[str] = None
    matched_keywords: str
    keyword_group: str = 'general'
    push_reason: str
    ai_confidence: float
    ai_summary: Optional[str] = None
    ai_key_points: str = ''
    ai_risk_level: Optional[str] = None
    ai_recommendation: Optional[str] = None
    source_name: Optional[str] = None
    source_platform: Optional[str] = None
    observed_at: Optional[str] = None
    source_count: int = 1
    cross_platform_count: int = 0
    spread_stage: Optional[str] = None
    statistical_note: Optional[str] = None
    forecast_note: Optional[str] = None
    risk_level: Optional[str] = None
    risk_type: Optional[str] = None
    event_summary: Optional[str] = None
    risk_narrative: Optional[str] = None
    impact_scope: Optional[str] = None
    recommendation: Optional[str] = None

    def __str__(self) -> str:
        if self.ai_summary:
            return f'{self.source_name or self.matched_keywords}\n{self.ai_summary}\n{self.url}'
        return f'{self.matched_keywords}\n{self.push_reason}\n{self.url}'

    def __repr__(self) -> str:
        return f'MiniMaxAnalysisRecord({self.video_id or self.post_id or self.url})'

    def get_uid(self) -> str:
        return self.video_id or self.post_id or self.url


@Plugins.register('minimax.analyze', Plugins.kind.ACTOR_CONFIG)
class MiniMaxAnalyzeConfig(QueueActionConfig):
    env_file: Path = Path('.env')
    """path to .env containing MiniMax settings"""
    api_key_env: Optional[str] = None
    """optional explicit API key env name. If omitted, international API URL prefers MINIMAX_INTL_API_KEY"""
    api_url_env: str = 'MINIMAX_API_URL'
    model_env: str = 'MINIMAX_MODEL'
    default_api_url: str = DEFAULT_API_URL
    default_model: str = DEFAULT_MODEL
    timeout: float = 60.0


@Plugins.register('minimax.analyze', Plugins.kind.ACTOR_ENTITY)
class MiniMaxAnalyzeEntity(QueueActionEntity):
    keywords_file: Path = Path('keywords.txt')
    """keyword file used to report matched keywords to the model and final notification"""
    keyword_group: str = 'general'
    """routing group copied into analysis records, e.g. general, hk, tw, apec"""
    min_confidence: float = Field(default=0.65, ge=0, le=1)
    """minimum model confidence required to push"""
    analysis_mode: str = 'risk'
    """risk keeps the original anti-China filter; profile_update pushes each new record; apec_risk adds structured APEC alert fields"""
    source_name: Optional[str] = None
    """human readable source name for profile_update notifications"""
    source_url: Optional[str] = None
    """source page url for profile_update notifications"""


@Plugins.register('minimax.analyze', Plugins.kind.ACTOR)
class MiniMaxAnalyzeAction(QueueAction):
    """
    Ask MiniMax whether a keyword-matched YouTube video has anti-China tendency.
    Only records confirmed by the model are forwarded.
    """

    def __init__(self, conf: MiniMaxAnalyzeConfig, entities: Sequence[MiniMaxAnalyzeEntity], ctx: RuntimeContext):
        super().__init__(conf, entities, ctx)
        self.conf: MiniMaxAnalyzeConfig
        self.entities: Dict[str, MiniMaxAnalyzeEntity]  # type: ignore
        self.env = load_env_file(conf.env_file, self.logger)
        self.api_url = self.env.get(conf.api_url_env, conf.default_api_url)
        self.model = self.env.get(conf.model_env, conf.default_model)
        if conf.api_key_env:
            self.api_key = self.env.get(conf.api_key_env, '')
        else:
            self.api_key = minimax_api_key_for(self.env, self.api_url)
        self.trust_env = minimax_trust_env_for(self.env, self.api_url)
        self.patterns_by_entity = {
            entity.name: load_patterns_file(entity.keywords_file, self.logger)
            for entity in entities
        }

    async def handle_single_record(self, logger: logging.Logger, client: HttpClient,
                                   entity: MiniMaxAnalyzeEntity, record: Record) -> None:
        try:
            analyzed = await self.analyze(entity, record)
        except Exception as e:
            logger.warning(f'[{entity.name}] MiniMax analysis failed for {record!r}: {type(e).__name__} {e}')
            return
        if analyzed is None:
            return
        self.on_record(entity, analyzed)

    async def analyze(self, entity: MiniMaxAnalyzeEntity, record: Record) -> Optional[Record]:
        if not self.api_key:
            self.logger.warning(f'[{entity.name}] MiniMax API key is missing, dropping record')
            return None

        keywords = matched_keywords(record, self.patterns_by_entity.get(entity.name, []))
        if entity.analysis_mode == 'profile_update':
            payload = self.build_profile_payload(entity, record)
        elif entity.analysis_mode == 'apec_risk':
            payload = self.build_apec_risk_payload(record, keywords)
        else:
            payload = self.build_payload(record, keywords)
        response = await asyncio.to_thread(self.call_minimax, payload)

        if entity.analysis_mode == 'profile_update':
            return self.profile_record(entity, record, response)

        should_push = bool(response.get('should_push'))
        confidence = float(response.get('confidence') or 0)
        reason = str(response.get('reason') or '').strip()
        if not should_push or confidence < entity.min_confidence:
            self.logger.debug(
                f'[{entity.name}] AI rejected record {record!r}: should_push={should_push}, confidence={confidence}, reason={reason}'
            )
            return None

        if entity.analysis_mode == 'apec_risk':
            return self.apec_risk_record(entity, record, keywords, response, reason, confidence)

        original = record.model_dump()
        observed_at = datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S UTC+08:00')
        return MiniMaxAnalysisRecord(
            title=original.get('title'),
            summary=original.get('summary'),
            author=original.get('author'),
            channel_link=original.get('channel_link'),
            url=original.get('url'),
            source_url=original.get('source_url'),
            video_id=original.get('video_id'),
            post_id=original.get('post_id'),
            published_text=original.get('published_text'),
            content=original.get('content'),
            matched_keywords='、'.join(keywords) if keywords else '未明确返回',
            keyword_group=entity.keyword_group,
            push_reason=reason,
            ai_confidence=confidence,
            source_name=entity.source_name or 'YouTube',
            source_platform='YouTube',
            observed_at=observed_at,
        )

    def apec_risk_record(self, entity: MiniMaxAnalyzeEntity, record: Record, keywords: Sequence[str],
                         response: Dict[str, Any], reason: str, confidence: float) -> MiniMaxAnalysisRecord:
        original = record.model_dump()
        risk_level = str(response.get('risk_level') or '关注').strip()
        risk_level = APEC_RISK_LEVELS.get(risk_level, APEC_RISK_LEVELS['关注'])
        risk_type = self.safe_analysis_text(response.get('risk_type'), '待核实')
        risk_narrative = self.safe_analysis_text(
            response.get('risk_narrative') or reason,
            '待核实（未获得可直接引用的风险依据）',
        )
        reason = self.safe_analysis_text(
            reason,
            '待核实（未获得可直接引用的风险依据）',
        )
        source_platform = entity.source_name or 'YouTube'
        observed_at = datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S UTC+08:00')

        return MiniMaxAnalysisRecord(
            title=original.get('title'),
            summary=original.get('summary'),
            author=original.get('author'),
            channel_link=original.get('channel_link'),
            url=original.get('url'),
            source_url=entity.source_url or original.get('source_url'),
            video_id=original.get('video_id'),
            post_id=original.get('post_id'),
            published_text=original.get('published_text'),
            content=original.get('content'),
            matched_keywords='、'.join(keywords) if keywords else 'APEC',
            keyword_group=entity.keyword_group,
            push_reason=reason,
            ai_confidence=confidence,
            source_name=source_platform,
            source_platform=source_platform,
            observed_at=observed_at,
            source_count=1,
            cross_platform_count=0,
            spread_stage='发现（单一 YouTube 来源）',
            statistical_note=(
                '当前 APEC 专项仅接入 YouTube；本预警由程序抓取到的 1 条视频触发，'
                '未采集跨平台讨论量、媒体跟进数或增长率。'
            ),
            forecast_note='暂无可验证的跨平台或时间序列统计，系统未生成未来趋势概率。',
            risk_level=risk_level,
            risk_type=risk_type,
            event_summary=original.get('title') or '待核实',
            risk_narrative=risk_narrative,
            impact_scope='当前仅确认 YouTube 公开搜索结果；其他平台、地区与对象未采集，待核实。',
            recommendation='建议人工核验原视频内容、频道信息及后续公开来源。',
        )

    @staticmethod
    def response_text(value: Any, fallback: str) -> str:
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            return ' / '.join(values) or fallback
        if value is None:
            return fallback
        text = str(value).strip()
        return text or fallback

    @classmethod
    def safe_analysis_text(cls, value: Any, fallback: str) -> str:
        text = cls.response_text(value, fallback)
        folded = text.casefold()
        if any(term.casefold() in folded for term in FORBIDDEN_ANALYSIS_TERMS):
            return fallback
        return text

    def profile_record(self, entity: MiniMaxAnalyzeEntity, record: Record, response: Dict[str, Any]) -> MiniMaxAnalysisRecord:
        original = record.model_dump()
        confidence = float(response.get('confidence') or 0)
        summary = str(response.get('summary') or '').strip()
        reason = str(response.get('reason') or response.get('analysis') or summary).strip()
        key_points = response.get('key_points') or response.get('keypoints') or []
        if isinstance(key_points, list):
            key_points_text = '\n'.join(f'- {item}' for item in key_points)
        else:
            key_points_text = str(key_points).strip()
        risk_level = str(response.get('risk_level') or response.get('risk') or '未知').strip()
        recommendation = str(response.get('recommendation') or response.get('suggestion') or '').strip()
        source_name = entity.source_name or original.get('author') or '重点账号动态'
        return MiniMaxAnalysisRecord(
            title=original.get('title'),
            summary=original.get('summary'),
            author=original.get('author'),
            channel_link=original.get('channel_link'),
            url=original.get('url'),
            source_url=entity.source_url or original.get('source_url'),
            video_id=original.get('video_id'),
            post_id=original.get('post_id'),
            published_text=original.get('published_text'),
            content=original.get('content'),
            matched_keywords=source_name,
            keyword_group=entity.keyword_group,
            push_reason=reason or summary,
            ai_confidence=confidence,
            ai_summary=summary or reason,
            ai_key_points=key_points_text,
            ai_risk_level=risk_level,
            ai_recommendation=recommendation,
            source_name=source_name,
        )

    def build_payload(self, record: Record, keywords: Sequence[str]) -> Dict[str, Any]:
        user_prompt = {
            'matched_keywords': list(keywords),
            'youtube_record': json.loads(record.as_json()),
            'content_excerpt': record_text(record),
        }
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是中文内容风险分析员。请判断给定 YouTube 视频是否不仅命中了关键词，'
                        '而且内容本身确实存在反华、辱华、分裂中国、煽动敌意或明显负面攻击中国/中国人的倾向。'
                        '只根据标题、频道、摘要、发布时间和链接等可见信息判断；证据不足时不要推送。'
                        '只输出 JSON 对象，不要输出额外文字。字段：'
                        'should_push(boolean), confidence(0到1), reason(中文，尽可能详细说明命中依据、语义判断和为什么应/不应推送)。'
                    ),
                },
                {'role': 'user', 'content': json.dumps(user_prompt, ensure_ascii=False)},
            ],
            'temperature': 0.1,
        }

    def build_profile_payload(self, entity: MiniMaxAnalyzeEntity, record: Record) -> Dict[str, Any]:
        user_prompt = {
            'source_name': entity.source_name,
            'source_url': entity.source_url,
            'youtube_record': json.loads(record.as_json()),
            'content_excerpt': record_text(record, limit=4000),
        }
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是重点账号动态监控分析员。请对给定 YouTube 社区动态进行中文分析。'
                        '这条动态只要是新内容就需要推送，不要做 should_push 过滤。'
                        '请输出 JSON 对象，不要输出额外文字。字段：'
                        'summary(中文摘要), key_points(字符串数组，提炼关键信息), '
                        'risk_level(低/中/高/未知，表示是否值得人工关注), '
                        'recommendation(中文，给出后续关注建议), '
                        'reason(中文，说明判断依据), confidence(0到1)。'
                    ),
                },
                {'role': 'user', 'content': json.dumps(user_prompt, ensure_ascii=False)},
            ],
            'temperature': 0.2,
        }

    def build_apec_risk_payload(self, record: Record, keywords: Sequence[str]) -> Dict[str, Any]:
        user_prompt = {
            'matched_keywords': list(keywords),
            'youtube_record': json.loads(record.as_json()),
            'content_excerpt': record_text(record),
        }
        return {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是 APEC 公开内容风险研判助手。先判断给定 YouTube 视频是否存在明确的反华、'
                        '辱华、分裂中国、煽动敌意或明显负面攻击中国/中国人的倾向；证据不足时不要推送。'
                        '只能依据输入中的标题、频道、摘要、发布时间、链接和内容摘录。'
                        '严禁虚构来源、发布时间、地点、人物、讨论量、增长率、媒体数量、跨平台传播、'
                        '传播阶段或未来趋势。输入未直接支持的内容必须写“待核实”。'
                        '不得提及任何大模型、云端服务或自身系统名称。'
                        '只输出 JSON 对象，不要输出额外文字。字段：'
                        'should_push(boolean), confidence(0到1), '
                        'risk_level(一般/关注/重要/重大，仅按可见证据分级), '
                        'risk_type(字符串数组，例如舆论/政治叙事/分裂言论；证据不足填待核实), '
                        'risk_narrative(中文，只说明输入中可直接看到的风险依据；'
                        '不得补充输入外事实，无直接证据填待核实), '
                        'reason(中文，说明为什么应或不应推送，必须指向可见依据)。'
                    ),
                },
                {'role': 'user', 'content': json.dumps(user_prompt, ensure_ascii=False)},
            ],
            'temperature': 0.1,
        }

    def call_minimax(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = request.Request(
            chat_url(self.api_url),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json',
            },
            data=data,
            method='POST',
        )
        if self.trust_env:
            openers = [request.build_opener()]
        else:
            openers = [
                request.build_opener(request.ProxyHandler({})),
                request.build_opener(),
            ]
        last_error: Optional[Exception] = None
        for opener in openers:
            try:
                with opener.open(req, timeout=self.conf.timeout) as response:
                    response_body = response.read().decode('utf-8')
                    status = response.status
                    break
            except Exception as e:
                last_error = e
        else:
            assert last_error is not None
            raise RuntimeError(f'MiniMax API request failed: {type(last_error).__name__} {last_error}') from last_error
        try:
            response_payload = json.loads(response_body)
        except ValueError as e:
            raise RuntimeError(f'MiniMax API returned non-JSON response: {response_body[:300]}') from e
        if status >= 400:
            raise RuntimeError(f'MiniMax API HTTP {status}: {response_payload}')
        message = extract_message(response_payload)
        return parse_json_object(message)

import asyncio
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from pydantic import AnyHttpUrl, Field, NonNegativeInt, PositiveFloat

from avtdl.core.actors import Actor, ActorConfig, ActorEntity
from avtdl.core.interfaces import Record
from avtdl.core.plugins import Plugins
from avtdl.plugins.mattermost.mattermost import load_env_file


CHINA_TZ = timezone(timedelta(hours=8))
LOG_TS_RE = re.compile(r'^(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')


def now_china() -> datetime:
    return datetime.now(CHINA_TZ)


def iso_china(dt: datetime) -> str:
    return dt.astimezone(CHINA_TZ).replace(microsecond=0).isoformat()


def utc_cutoff(seconds: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')


def parse_db_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_log_timestamp(line: str) -> Optional[datetime]:
    match = LOG_TS_RE.match(line)
    if match is None:
        return None
    try:
        naive = datetime.strptime(match.group('ts'), '%Y/%m/%d %H:%M:%S.%f')
    except ValueError:
        return None
    return naive.replace(tzinfo=CHINA_TZ)


@Plugins.register('kan.status_report', Plugins.kind.ACTOR_CONFIG)
class KanStatusReportConfig(ActorConfig):
    base_url: AnyHttpUrl = 'https://kan.cool'
    """KAN base URL"""
    endpoint_path: str = '/api/external/agent-status/report'
    """KAN status report endpoint path"""
    env_file: Path = Path('.env')
    """path to .env containing status report credentials"""
    report_key_env: str = 'YOUTUBE_STATUS_REPORT_KEY'
    """env var containing X-Agent-Report-Key"""
    secret_env: str = 'KAN_AGENT_STATUS_REPORT_SECRET'
    """env var containing Bearer token for status reports"""
    db_path: Path = Path('runtime/youtube/db')
    """SQLite records database used for collection statistics"""
    state_file: Path = Path('runtime/youtube/state/kan_status_report.json')
    """local JSON state file used to persist scan_round"""
    log_files: List[Path] = Field(default_factory=lambda: [Path('log/avtdl.log')])
    """log files scanned for recent failure counters"""
    report_interval: PositiveFloat = 600
    """seconds between status reports"""
    stale_after_seconds: NonNegativeInt = 900
    """seconds after which KAN should treat the status as stale"""
    report_window_seconds: NonNegativeInt = 86400
    """time window used for counters in the status payload"""
    request_timeout: PositiveFloat = 20
    """HTTP timeout in seconds"""


@Plugins.register('kan.status_report', Plugins.kind.ACTOR_ENTITY)
class KanStatusReportEntity(ActorEntity):
    feed_names: List[str] = Field(
        default_factory=lambda: [
            'youtube keyword search',
            'youtube APEC search',
            'jintsumi community posts',
        ]
    )
    """all feeds included in total status counters"""
    normal_feed_names: List[str] = Field(default_factory=lambda: ['youtube keyword search'])
    """feeds counted as normal YouTube monitoring"""
    apec_feed_names: List[str] = Field(default_factory=lambda: ['youtube APEC search'])
    """feeds counted as APEC monitoring"""


@Plugins.register('kan.status_report', Plugins.kind.ACTOR)
class KanStatusReportActor(Actor):
    def __init__(self, conf: KanStatusReportConfig, entities: Sequence[KanStatusReportEntity], ctx):
        super().__init__(conf, entities, ctx)
        self.conf: KanStatusReportConfig
        self.entities: Dict[str, KanStatusReportEntity]  # type: ignore
        self.env: Dict[str, str] = {}

    def handle_record(self, entity: KanStatusReportEntity, record: Record) -> None:
        return

    async def run(self) -> None:
        self.env = load_env_file(self.conf.env_file, self.logger)
        if not self.report_key() or not self.secret():
            self.logger.warning(
                f'status report credentials are missing in "{self.conf.env_file}", '
                f'expected "{self.conf.report_key_env}" and "{self.conf.secret_env}"'
            )
            return
        for entity in self.entities.values():
            task_name = f'{self.conf.name}:{entity.name}'
            self.controller.create_task(self.run_for(entity), name=task_name)
        await super().run()

    async def run_for(self, entity: KanStatusReportEntity) -> None:
        while True:
            try:
                await self.report_once(entity)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception(f'[{entity.name}] failed to report KAN status')
            await asyncio.sleep(float(self.conf.report_interval))

    def report_key(self) -> str:
        return self.env.get(self.conf.report_key_env, '').strip()

    def secret(self) -> str:
        return self.env.get(self.conf.secret_env, '').strip()

    def endpoint_url(self) -> str:
        base = str(self.conf.base_url).rstrip('/')
        path = self.conf.endpoint_path
        if not path.startswith('/'):
            path = f'/{path}'
        return f'{base}{path}'

    async def report_once(self, entity: KanStatusReportEntity) -> None:
        payload = self.build_payload(entity)
        headers = {
            'Content-Type': 'application/json',
            'X-Agent-Report-Key': self.report_key(),
            'Authorization': f'Bearer {self.secret()}',
        }
        timeout = aiohttp.ClientTimeout(total=float(self.conf.request_timeout))
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.endpoint_url(), headers=headers, json=payload) as response:
                text = await response.text()
                if 200 <= response.status < 300:
                    self.logger.info(
                        f'[{entity.name}] reported KAN status: '
                        f'status={response.status}, scan_round={payload["scan_round"]}, '
                        f'candidate_post_count={payload["candidate_post_count"]}'
                    )
                    return
                safe_text = text[:500].replace('\n', ' ')
                self.logger.warning(
                    f'[{entity.name}] KAN status report failed: {response.status} {response.reason} {safe_text}'
                )

    def build_payload(self, entity: KanStatusReportEntity) -> Dict[str, Any]:
        stats = self.collect_db_stats(entity)
        log_stats = self.collect_log_stats()
        now = now_china()
        payload = {
            'reported_at': iso_china(now),
            'last_scan_at': iso_china(stats['last_scan_at']) if stats['last_scan_at'] else None,
            'period_started_at': iso_china(now - timedelta(seconds=self.conf.report_window_seconds)),
            'stale_after_seconds': int(self.conf.stale_after_seconds),
            'scan_round': self.next_scan_round(),
            'scan_success_count': 1 if stats['last_scan_at'] is not None else 0,
            'candidate_post_count': stats['candidate_post_count'],
            'ai_approved_count': stats['ai_approved_count'],
            'ai_filtered_count': stats['ai_filtered_count'],
            'ai_analysis_error_count': log_stats['ai_analysis_error_count'],
            'published_count': log_stats['published_count'],
            'published_normal_count': log_stats['published_normal_count'],
            'published_apec_count': log_stats['published_apec_count'],
            'fetch_failure_count': log_stats['fetch_failure_count'],
            'publish_failure_count': log_stats['publish_failure_count'],
        }
        return payload

    def collect_db_stats(self, entity: KanStatusReportEntity) -> Dict[str, Any]:
        counters = {
            'candidate_post_count': 0,
            'ai_approved_count': 0,
            'ai_filtered_count': 0,
            'last_scan_at': None,
        }
        if not self.conf.db_path.exists():
            self.logger.warning(f'record database "{self.conf.db_path}" does not exist')
            return counters

        all_feeds = set(entity.feed_names)
        placeholders = ','.join('?' for _ in all_feeds)
        if not placeholders:
            return counters

        try:
            conn = sqlite3.connect(str(self.conf.db_path))
            try:
                cursor = conn.execute(
                    f'select parsed_at, feed_name, as_json from records '
                    f'where feed_name in ({placeholders}) and parsed_at >= ?',
                    [*all_feeds, utc_cutoff(int(self.conf.report_window_seconds))],
                )
                for parsed_at, feed_name, as_json in cursor:
                    counters['candidate_post_count'] += 1
                    self.update_ai_counters(counters, as_json)
                    parsed_dt = parse_db_timestamp(parsed_at)
                    last_dt = counters['last_scan_at']
                    if parsed_dt is not None and (last_dt is None or parsed_dt > last_dt):
                        counters['last_scan_at'] = parsed_dt

                latest = conn.execute(
                    f'select max(parsed_at) from records where feed_name in ({placeholders})',
                    [*all_feeds],
                ).fetchone()
                latest_dt = parse_db_timestamp(latest[0] if latest else None)
                if latest_dt is not None:
                    counters['last_scan_at'] = latest_dt
            finally:
                conn.close()
        except sqlite3.Error as e:
            self.logger.warning(f'failed to collect status stats from "{self.conf.db_path}": {e}')
        return counters

    @staticmethod
    def update_ai_counters(counters: Dict[str, Any], as_json: object) -> None:
        try:
            data = json.loads(str(as_json))
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return
        confidence = data.get('ai_confidence')
        if confidence is None:
            return
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return
        if value >= 0.65:
            counters['ai_approved_count'] += 1
        else:
            counters['ai_filtered_count'] += 1

    def collect_log_stats(self) -> Dict[str, int]:
        counters = {
            'ai_analysis_error_count': 0,
            'published_count': 0,
            'published_normal_count': 0,
            'published_apec_count': 0,
            'fetch_failure_count': 0,
            'publish_failure_count': 0,
        }
        cutoff = now_china() - timedelta(seconds=self.conf.report_window_seconds)
        for path in self.conf.log_files:
            self.scan_log_file(path, cutoff, counters)
        return counters

    def scan_log_file(self, path: Path, cutoff: datetime, counters: Dict[str, int]) -> None:
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except FileNotFoundError:
            return
        except OSError as e:
            self.logger.warning(f'failed to read log file "{path}": {e}')
            return
        for line in lines[-5000:]:
            dt = parse_log_timestamp(line)
            if dt is not None and dt < cutoff:
                continue
            lower = line.lower()
            if 'sent record to mattermost channel' in lower:
                counters['published_count'] += 1
                if 'apec keyword group' in lower:
                    counters['published_apec_count'] += 1
                elif 'kan keyword groups' in lower or 'yt ' in lower:
                    counters['published_normal_count'] += 1
            if 'failed to send record to mattermost' in lower or 'kan status report failed' in lower:
                counters['publish_failure_count'] += 1
            if 'minimax' in lower and (' failed' in lower or ' error' in lower or 'exception' in lower):
                counters['ai_analysis_error_count'] += 1
            if (
                ('actor.channel' in lower or 'youtube.community' in lower or 'youtube' in lower)
                and ('failed' in lower or 'error' in lower or 'exception' in lower)
            ):
                counters['fetch_failure_count'] += 1

    def next_scan_round(self) -> int:
        state = self.load_state()
        scan_round = int(state.get('scan_round') or 0) + 1
        state['scan_round'] = scan_round
        self.store_state(state)
        return scan_round

    def load_state(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.conf.state_file.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f'failed to load KAN status state from "{self.conf.state_file}": {e}')
            return {}
        return data if isinstance(data, dict) else {}

    def store_state(self, state: Dict[str, Any]) -> None:
        try:
            self.conf.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.conf.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError as e:
            self.logger.warning(f'failed to store KAN status state to "{self.conf.state_file}": {e}')

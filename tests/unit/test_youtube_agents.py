import json
import sqlite3
from datetime import datetime, timezone

from avtdl.core.config import SettingsSection
from avtdl.core.interfaces import OpaqueRecord
from avtdl.core.runtime import MessageBus, RuntimeContext, TasksController
from avtdl.plugins.minimax.analyze import MiniMaxAnalyzeAction, MiniMaxAnalyzeConfig, MiniMaxAnalyzeEntity
from avtdl.plugins.mattermost.mattermost import MattermostAction, MattermostConfig, MattermostEntity
from avtdl.plugins.youtube.agents import (
    YouTubeKeywordAgentEntity,
    YouTubeKeywordAgentsAction,
    YouTubeKeywordAgentsConfig,
)


def make_context(tmp_path):
    ctx = RuntimeContext(MessageBus(), TasksController())
    ctx.set_extra('settings', SettingsSection(cache_directory=tmp_path / 'cache'))
    return ctx


def make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        'create table records (parsed_at datetime, feed_name text, uid text, hashsum text, class_name text, as_json text, primary key(uid, hashsum))'
    )
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f+00:00')
    rows = [
        ('youtube keyword search', '1', {'author': 'Channel A', 'title': 'first'}),
        ('youtube keyword search', '2', {'author': 'Channel B', 'title': 'second'}),
        ('youtube APEC search', '3', {'author': 'APEC Channel', 'title': 'other feed'}),
    ]
    for feed, uid, payload in rows:
        conn.execute(
            'insert into records values (?, ?, ?, ?, ?, ?)',
            (now, feed, uid, uid, 'YoutubeVideoRecord', json.dumps(payload, ensure_ascii=False)),
        )
    conn.commit()
    conn.close()


def make_record():
    return OpaqueRecord(
        title='Test title',
        author='Channel A',
        channel_link='https://www.youtube.com/@channel-a',
        url='https://www.youtube.com/watch?v=abc',
        video_id='abc',
        published_text='2 hours ago',
        matched_keywords='反华',
        push_reason='标题和摘要出现明确负面攻击中国的表述。',
        ai_confidence=0.9,
    )


def test_collector_agent_uses_real_recent_stats(tmp_path):
    db_path = tmp_path / 'records.sqlite'
    make_db(db_path)
    action = YouTubeKeywordAgentsAction(
        YouTubeKeywordAgentsConfig(
            name='youtube.keyword_agents',
            db_path=db_path,
            stats_window_hours=24,
            baseline_window_hours=24,
        ),
        [YouTubeKeywordAgentEntity(name='collector', agent_role='collector')],
        make_context(tmp_path),
    )

    output = action.build_agent_record(action.entities['collector'], make_record())

    assert output.agent_role == 'collector'
    assert output.raw_collected_rows == 2
    assert output.account_count == 2
    assert output.related_accounts == 'Channel A、Channel B'
    assert output.risk_level == '高'
    assert '近 24 小时' in output.collector_note


def test_key_account_matching_uses_account_file(tmp_path):
    accounts = tmp_path / 'key_accounts.youtube.txt'
    accounts.write_text('@channel-a\n', encoding='utf-8')
    action = YouTubeKeywordAgentsAction(
        YouTubeKeywordAgentsConfig(name='youtube.keyword_agents', db_path=tmp_path / 'missing.sqlite'),
        [YouTubeKeywordAgentEntity(name='key', agent_role='key_accounts', key_accounts_file=accounts)],
        make_context(tmp_path),
    )

    assert action.is_key_account(action.entities['key'], make_record())


def test_agent_record_preserves_keyword_group(tmp_path):
    action = YouTubeKeywordAgentsAction(
        YouTubeKeywordAgentsConfig(name='youtube.keyword_agents', db_path=tmp_path / 'missing.sqlite'),
        [YouTubeKeywordAgentEntity(name='collector', agent_role='collector')],
        make_context(tmp_path),
    )
    record = OpaqueRecord(
        title='Hong Kong test',
        author='Channel A',
        url='https://www.youtube.com/watch?v=hk',
        video_id='hk',
        matched_keywords='港独',
        keyword_group='hk',
        push_reason='matched HK keyword',
        ai_confidence=0.8,
    )

    output = action.build_agent_record(action.entities['collector'], record)

    assert output.keyword_group == 'hk'


def test_agent_record_preserves_ai_route_group(tmp_path):
    action = YouTubeKeywordAgentsAction(
        YouTubeKeywordAgentsConfig(name='youtube.keyword_agents', db_path=tmp_path / 'missing.sqlite'),
        [YouTubeKeywordAgentEntity(name='collector', agent_role='collector')],
        make_context(tmp_path),
    )
    record = OpaqueRecord(
        title='SVIP routed test',
        author='Channel A',
        url='https://www.youtube.com/watch?v=svip',
        video_id='svip',
        matched_keywords='反华',
        keyword_group='general',
        ai_route_group='svip',
        push_reason='AI recommends SVIP routing',
        ai_confidence=0.8,
    )

    output = action.build_agent_record(action.entities['collector'], record)

    assert output.keyword_group == 'general'
    assert output.ai_route_group == 'svip'


def test_mattermost_routes_channels_by_keyword_group(tmp_path):
    action = MattermostAction(
        MattermostConfig(name='mattermost'),
        [
            MattermostEntity(
                name='keyword groups',
                channels=['general-a', 'general-b'],
                channel_routes={
                    'keyword_group': {
                        'hk': ['a1'],
                        'tw': ['a2'],
                        'general': ['general-a', 'general-b'],
                    }
                },
            )
        ],
        make_context(tmp_path),
    )
    entity = action.entities['keyword groups']

    assert action.channels_for(entity, OpaqueRecord(keyword_group='hk', url='https://example.com/hk')) == ['a1']
    assert action.channels_for(entity, OpaqueRecord(keyword_group='tw', url='https://example.com/tw')) == ['a2']
    assert action.channels_for(entity, OpaqueRecord(keyword_group='general', url='https://example.com/general')) == [
        'general-a',
        'general-b',
    ]


def test_mattermost_prefers_ai_route_group_with_keyword_group_fallback(tmp_path):
    action = MattermostAction(
        MattermostConfig(name='mattermost'),
        [
            MattermostEntity(
                name='keyword groups',
                channels=['general-a', 'general-b'],
                channel_routes={
                    'ai_route_group': {
                        'svip': ['svip'],
                        'hk': ['a1'],
                        'tw': ['a2'],
                    },
                    'keyword_group': {
                        'hk': ['a1'],
                        'tw': ['a2'],
                        'general': ['general-a', 'general-b'],
                    },
                },
            )
        ],
        make_context(tmp_path),
    )
    entity = action.entities['keyword groups']

    assert action.channels_for(
        entity,
        OpaqueRecord(keyword_group='general', ai_route_group='svip', url='https://example.com/svip'),
    ) == ['svip']
    assert action.channels_for(
        entity,
        OpaqueRecord(keyword_group='tw', ai_route_group='unknown', url='https://example.com/fallback'),
    ) == ['a2']


def test_apec_requires_direct_negative_china_evidence(tmp_path):
    action = MiniMaxAnalyzeAction(
        MiniMaxAnalyzeConfig(name='minimax.analyze', env_file=tmp_path / 'missing.env'),
        [
            MiniMaxAnalyzeEntity(
                name='apec',
                keywords_file=tmp_path / 'missing-keywords.txt',
                keyword_group='apec',
                analysis_mode='apec_risk',
            )
        ],
        make_context(tmp_path),
    )

    assert not action.has_direct_negative_evidence('无')
    assert not action.has_direct_negative_evidence('待核实')
    assert action.has_direct_negative_evidence('标题直接指责中国在APEC中破坏地区秩序')

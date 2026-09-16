import json
import sqlite3
from datetime import datetime, timezone

from avtdl.core.runtime import MessageBus, RuntimeContext, TasksController
from avtdl.plugins.kan.status_report import (
    KanStatusReportActor,
    KanStatusReportConfig,
    KanStatusReportEntity,
)


def make_context():
    return RuntimeContext(MessageBus(), TasksController())


def make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        'create table records (parsed_at datetime, feed_name text, uid text, hashsum text, class_name text, as_json text, primary key(uid, hashsum))'
    )
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f+00:00')
    rows = [
        ('youtube keyword search', '1', {'title': 'normal'}),
        ('youtube APEC search', '2', {'title': 'apec', 'ai_confidence': 0.9}),
        ('jintsumi community posts', '3', {'title': 'community', 'ai_confidence': 0.2}),
    ]
    for feed, uid, payload in rows:
        conn.execute(
            'insert into records values (?, ?, ?, ?, ?, ?)',
            (now, feed, uid, uid, 'OpaqueRecord', json.dumps(payload, ensure_ascii=False)),
        )
    conn.commit()
    conn.close()


def test_status_payload_uses_db_and_log_counters(tmp_path):
    db_path = tmp_path / 'records.sqlite'
    log_path = tmp_path / 'avtdl.log'
    state_file = tmp_path / 'state.json'
    make_db(db_path)
    log_path.write_text(
        '\n'.join(
            [
                '2026/09/16 10:00:00.000 [DEBUG] [actor.mattermost] [kan keyword groups] sent record to Mattermost channel "normal"',
                '2026/09/16 10:00:01.000 [DEBUG] [actor.mattermost] [apec keyword group] sent record to Mattermost channel "apec"',
                '2026/09/16 10:00:02.000 [WARNING] [actor.mattermost] [apec keyword group] failed to send record to Mattermost channel "apec"',
                '2026/09/16 10:00:03.000 [ERROR] [actor.minimax.analyze] minimax request failed',
            ]
        ),
        encoding='utf-8',
    )

    actor = KanStatusReportActor(
        KanStatusReportConfig(
            name='kan.status_report',
            db_path=db_path,
            state_file=state_file,
            log_files=[log_path],
            report_window_seconds=86400,
        ),
        [KanStatusReportEntity(name='youtube hk status')],
        make_context(),
    )

    payload = actor.build_payload(actor.entities['youtube hk status'])

    assert payload['scan_round'] == 1
    assert payload['scan_success_count'] == 1
    assert payload['candidate_post_count'] == 3
    assert payload['ai_approved_count'] == 1
    assert payload['ai_filtered_count'] == 1
    assert payload['published_count'] == 2
    assert payload['published_normal_count'] == 1
    assert payload['published_apec_count'] == 1
    assert payload['publish_failure_count'] == 1
    assert payload['ai_analysis_error_count'] == 1
    assert payload['last_scan_at'] is not None

    second_payload = actor.build_payload(actor.entities['youtube hk status'])
    assert second_payload['scan_round'] == 2

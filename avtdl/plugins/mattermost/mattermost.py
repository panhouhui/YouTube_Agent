import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from pydantic import AnyHttpUrl, Field

from avtdl.core.actions import QueueAction, QueueActionConfig, QueueActionEntity
from avtdl.core.formatters import Fmt
from avtdl.core.interfaces import Record
from avtdl.core.plugins import Plugins
from avtdl.core.request import HttpClient, RequestDetails, RetrySettings
from avtdl.core.runtime import RuntimeContext


DEFAULT_TOKEN_ENV = 'MATTERMOST_BOT_TOKEN'


def load_env_file(path: Path, logger: logging.Logger) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding='utf8').splitlines()
    except FileNotFoundError:
        logger.warning(f'env file "{path}" does not exist')
        return values
    except OSError as e:
        logger.warning(f'failed to read env file "{path}": {e}')
        return values

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if '=' not in line:
            logger.warning(f'skipping malformed line {line_number} in env file "{path}"')
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


@Plugins.register('mattermost', Plugins.kind.ACTOR_CONFIG)
class MattermostConfig(QueueActionConfig):
    base_url: AnyHttpUrl = 'https://kan.cool'
    """Mattermost server base url"""
    env_file: Path = Path('.env')
    """path to a .env file containing the bot token"""
    token_env: str = DEFAULT_TOKEN_ENV
    """environment variable name used to read the bot token from env_file"""
    sent_history_file: Path = Path('runtime/youtube/state/mattermost_sent.json')
    """path to a JSON file containing already sent record ids"""


@Plugins.register('mattermost', Plugins.kind.ACTOR_ENTITY)
class MattermostEntity(QueueActionEntity):
    channels: List[str] = Field(default_factory=list)
    """Mattermost channel ids to post into"""
    channel_env: Optional[str] = None
    """optional env var containing one Mattermost channel id"""
    channel_envs: List[str] = Field(default_factory=list)
    """optional env vars containing Mattermost channel ids"""
    token_env: Optional[str] = None
    """optional env var containing a bot token for this entity"""
    required_fields: Dict[str, str] = Field(default_factory=dict)
    """optional record field values required before this entity sends a message"""
    message_template: Optional[str] = None
    """optional template used to format the message. If omitted, record text representation is used"""


@Plugins.register('mattermost', Plugins.kind.ACTOR)
class MattermostAction(QueueAction):
    """
    Send records to Mattermost channels

    Posts incoming records to one or more Mattermost channels using the REST API.
    The bot token is read from a local .env file instead of being stored in the
    avtdl configuration file.
    """

    def __init__(self, conf: MattermostConfig, entities: Sequence[MattermostEntity], ctx: RuntimeContext):
        super().__init__(conf, entities, ctx)
        self.conf: MattermostConfig
        self.entities: Dict[str, MattermostEntity]  # type: ignore
        self.token: Optional[str] = None
        self.env: Dict[str, str] = {}
        self.tokens: Dict[str, str] = {}
        self.sent_history: Set[str] = set()

    def load_token(self, token_env: str) -> Optional[str]:
        token = self.env.get(token_env)
        if not token:
            self.logger.warning(
                f'token "{token_env}" was not found in env file "{self.conf.env_file}"'
            )
            return None
        return token

    async def run(self) -> None:
        self.env = load_env_file(self.conf.env_file, self.logger)
        self.token = self.load_token(self.conf.token_env)
        self.sent_history = self.load_sent_history()
        await super().run()

    def load_sent_history(self) -> Set[str]:
        path = self.conf.sent_history_file
        try:
            data = json.loads(path.read_text(encoding='utf8'))
        except FileNotFoundError:
            return set()
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f'failed to load Mattermost sent history from "{path}": {e}')
            return set()
        if not isinstance(data, list):
            self.logger.warning(f'failed to load Mattermost sent history from "{path}": expected JSON list')
            return set()
        return {str(item) for item in data}

    def store_sent_history(self) -> None:
        path = self.conf.sent_history_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sorted(self.sent_history), ensure_ascii=False, indent=2), encoding='utf8')
        except OSError as e:
            self.logger.warning(f'failed to store Mattermost sent history to "{path}": {e}')

    def sent_key(self, record: Record) -> str:
        video_id = getattr(record, 'video_id', None)
        if video_id:
            return f'video:{video_id}'
        url = getattr(record, 'url', None)
        if url:
            return f'url:{url}'
        return f'hash:{record.hash()}'

    def endpoint_url(self) -> str:
        base = str(self.conf.base_url).rstrip('/')
        return f'{base}/api/v4/posts'

    def token_for(self, entity: MattermostEntity) -> Optional[str]:
        token_env = entity.token_env or self.conf.token_env
        if token_env == self.conf.token_env:
            return self.token
        if token_env not in self.tokens:
            token = self.load_token(token_env)
            if token is not None:
                self.tokens[token_env] = token
        return self.tokens.get(token_env)

    def channels_for(self, entity: MattermostEntity) -> List[str]:
        channels = list(entity.channels)
        env_names = list(entity.channel_envs)
        if entity.channel_env:
            env_names.append(entity.channel_env)
        for env_name in env_names:
            channel_id = self.env.get(env_name)
            if channel_id:
                channels.append(channel_id)
            else:
                self.logger.warning(
                    f'channel id "{env_name}" was not found in env file "{self.conf.env_file}"'
                )
        return channels

    def prepare_message(self, entity: MattermostEntity, record: Record) -> str:
        if entity.message_template is None:
            return str(record)
        return Fmt.format(entity.message_template, record, tz=entity.timezone)

    def record_matches(self, entity: MattermostEntity, record: Record) -> bool:
        if not entity.required_fields:
            return True
        data = record.model_dump()
        for field, expected in entity.required_fields.items():
            value = data.get(field)
            if str(value) != str(expected):
                return False
        return True

    async def handle_single_record(self, logger: logging.Logger, client: HttpClient,
                                   entity: MattermostEntity, record: Record) -> None:
        if not self.record_matches(entity, record):
            logger.debug(f'[{entity.name}] record does not match required fields, skipping')
            return
        token = self.token_for(entity)
        if token is None:
            logger.warning(f'[{entity.name}] Mattermost bot token is not configured, skipping record')
            return
        channels = self.channels_for(entity)
        if not channels:
            logger.warning(f'[{entity.name}] Mattermost channels are not configured, skipping record')
            return

        key = f'{entity.name}:{self.sent_key(record)}'
        if key in self.sent_history:
            logger.info(f'[{entity.name}] record "{key}" was already sent to Mattermost, skipping duplicate')
            return

        message = self.prepare_message(entity, record)
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json; charset=utf-8'
        }

        sent_count = 0
        for channel_id in channels:
            payload = {
                'channel_id': channel_id,
                'message': message
            }
            request = RequestDetails(
                url=self.endpoint_url(),
                method='POST',
                data_json=payload,
                headers=headers,
                retry_settings=RetrySettings(retry_times=2, retry_delay=2)
            )
            response = await client.request_endpoint(logger, request)
            if response.ok:
                sent_count += 1
                logger.debug(f'[{entity.name}] sent record to Mattermost channel "{channel_id}"')
                continue
            logger.warning(
                f'[{entity.name}] failed to send record to Mattermost channel "{channel_id}": '
                f'{response.status} {response.reason} {response.text}'
            )
        if sent_count == len(channels):
            self.sent_history.add(key)
            self.store_sent_history()
